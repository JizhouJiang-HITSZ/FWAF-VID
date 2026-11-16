import contextlib
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'script'))
import bag2tum
import file_utils
import indoor_bag2gt as indoor
import outdoor_bag2gt as outdoor
import kalman_fillter as kalman
import rosbag2video as video


class FilesAndTrajectories(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bag = self.root / 'bag with spaces & symbols.bag'
        self.bag.write_bytes(b'input')
        self.output = self.root / 'result.tum'

    def test_atomic_failure_preserves_existing_output_and_cleans_temp(self):
        self.output.write_text('original')
        with self.assertRaises(RuntimeError):
            with file_utils.atomic_output(self.output) as temporary:
                temporary.write_text('partial')
                raise RuntimeError('export failed')
        self.assertEqual(self.output.read_text(), 'original')
        self.assertEqual(set(self.root.iterdir()), {self.bag, self.output})

    def test_input_output_collision_rejected(self):
        with self.assertRaises(ValueError):
            file_utils.validate_paths(self.bag, self.bag)
        self.assertEqual(self.bag.read_bytes(), b'input')

    def test_rostopic_receives_literal_arguments_and_failure_propagates(self):
        with patch('file_utils.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'rostopic')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                file_utils.export_topic_csv(self.bag, '/topic;literal', self.root / 'export.csv')
        self.assertEqual(run.call_args.args[0],
                         ['rostopic', 'echo', '-b', str(self.bag.resolve()), '-p', '/topic;literal'])
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertNotIn('shell', run.call_args.kwargs)

    def indoor_export(self, bag, topic, target):
        position = np.array([0.25, 0.5])
        anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        row = {'%time': 1000000000, 'field.pos_3d2': 2.5}
        row.update({f'field.dis_arr{i}': d for i, d in enumerate(np.linalg.norm(anchors-position, axis=1))})
        row.update({f'field.quaternion{i}': q for i, q in enumerate([0, 0, 0, 1])})
        pd.DataFrame([row]).to_csv(target, index=False)

    def test_indoor_repeat_does_not_duplicate_and_geometry_is_unchanged(self):
        with patch.object(indoor, 'export_topic_csv', side_effect=self.indoor_export):
            indoor.convert_to_tum_with_toa(self.bag, self.output, '/uwb')
            first = self.output.read_bytes()
            indoor.convert_to_tum_with_toa(self.bag, self.output, '/uwb')
        self.assertEqual(first, self.output.read_bytes())
        np.testing.assert_allclose(np.loadtxt(self.output), [1, .25, .5, 2.5, 0, 0, 0, 1], atol=1e-6)

    def test_failed_indoor_conversion_preserves_output(self):
        self.output.write_text('original')
        with patch.object(indoor, 'export_topic_csv', side_effect=self.indoor_export), \
             patch.object(indoor, 'fit_toa', side_effect=ValueError('bad distances')):
            with self.assertRaises(ValueError):
                indoor.convert_to_tum_with_toa(self.bag, self.output, '/uwb')
        self.assertEqual(self.output.read_text(), 'original')

    def outdoor_export(self, bag, topic, target, imu_count=6):
        if topic == '/rtk':
            frame = pd.DataFrame({'%time': [1000000000, 2000000000],
                                  'field.latitude': [0, .001], 'field.longitude': [0, 0],
                                  'field.altitude': [10, 12]})
        else:
            frame = pd.DataFrame({f'field.orientation.{axis}':
                                  [float(axis == 'w') for i in range(imu_count)] for axis in 'xyzw'})
            if imu_count >= 6:
                frame.loc[5, 'field.orientation.z'] = 1.0
                frame.loc[5, 'field.orientation.w'] = 0.0
        frame.to_csv(target, index=False)

    def test_outdoor_existing_formula_and_five_to_one_association(self):
        with patch.object(outdoor, 'export_topic_csv', side_effect=self.outdoor_export):
            outdoor.convert_to_tum(self.bag, self.output, '/rtk', '/imu')
        data = np.loadtxt(self.output)
        np.testing.assert_allclose(data[0], [1, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(data[1], [2, 111.319700232, 0, 2, 0, 0, 1, 0], atol=1e-6)

    def test_short_imu_fails_without_touching_output(self):
        self.output.write_text('original')
        def exporter(bag, topic, target):
            self.outdoor_export(bag, topic, target, imu_count=5)
        with patch.object(outdoor, 'export_topic_csv', side_effect=exporter):
            with self.assertRaisesRegex(ValueError, '5\*i'):
                outdoor.convert_to_tum(self.bag, self.output, '/rtk', '/imu')
        self.assertEqual(self.output.read_text(), 'original')

    def test_default_filenames_remove_only_suffix(self):
        with patch.object(indoor, 'convert_to_tum_with_toa') as convert:
            indoor.main(['--bag', 'bag.bag'])
            self.assertEqual(convert.call_args.args[1], 'bag_toa_fit.tum')
        with patch.object(outdoor, 'convert_to_tum') as convert:
            outdoor.main(['-b', 'bag.bag'])
            self.assertEqual(convert.call_args.args[1], 'bag.tum')

    def test_filter_whitespace_and_unmodified_z_quaternion(self):
        source = self.root / 'trajectory.txt'
        source.write_text('1  0\t0 3 0 0 0 1\n2  2\t4 5 0 0 1 0\n')
        kalman.main(source, self.output)
        data = np.loadtxt(self.output)
        np.testing.assert_array_equal(data[:, [0, 3, 4, 5, 6, 7]],
                                      [[1, 3, 0, 0, 0, 1], [2, 5, 0, 0, 1, 0]])

    def test_all_script_help_and_missing_required_arguments(self):
        for name in ['bag2tum', 'indoor_bag2gt', 'outdoor_bag2gt', 'kalman_fillter',
                     'cal_optical_flow', 'rosbag2video']:
            script = str(ROOT / 'script' / (name + '.py'))
            with self.subTest(script=name):
                help_result = subprocess.run([sys.executable, script, '--help'], capture_output=True)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                missing = subprocess.run([sys.executable, script], capture_output=True)
                self.assertEqual(missing.returncode, 2, missing.stderr)
                self.assertNotIn(b'Traceback', missing.stderr)


class FakeBag:
    messages = []
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read_messages(self, **kwargs):
        for topic, msg, time in self.messages:
            if 'topics' in kwargs and topic not in kwargs['topics']:
                continue
            if 'connection_filter' in kwargs:
                datatype = 'sensor_msgs/CompressedImage' if hasattr(msg, 'format') else 'sensor_msgs/Image'
                if not kwargs['connection_filter'](topic, datatype, None, None, None):
                    continue
            yield topic, msg, time


class FakePipe(io.BytesIO):
    def close(self): self.saved = self.getvalue(); super().close()


class FakeEncoder:
    instances = []
    code = 0
    def __init__(self, command, **kwargs):
        self.stdin = FakePipe()
        self.target = Path(command[-1])
        self.instances.append(self)
    def wait(self, timeout=None):
        self.target.write_bytes(b'encoded')
        return self.code
    def kill(self): pass


def raw_frame(value=0, width=2, height=2):
    return SimpleNamespace(data=bytes([value]) * (width * height), encoding='mono8',
                           width=width, height=height, step=width, is_bigendian=False)


def stamp(seconds):
    return SimpleNamespace(to_sec=lambda: seconds)


class VideoExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bag = self.root / 'sample.bag'
        self.bag.write_bytes(b'bag')
        self.output = self.root / 'result.mp4'
        FakeBag.messages = []
        FakeEncoder.instances, FakeEncoder.code = [], 0
        self.ros = patch.dict(sys.modules, {'rosbag': SimpleNamespace(Bag=FakeBag)})
        self.ros.start()
        self.addCleanup(self.ros.stop)

    def writer(self, **kwargs):
        return video.RosVideoWriter(topic='/camera', output_filename=str(self.output), **kwargs)

    def test_one_frame_is_written_and_encoder_is_closed(self):
        FakeBag.messages = [('/camera', raw_frame(7), stamp(0))]
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            self.writer().addBag(self.bag)
        self.assertEqual(FakeEncoder.instances[0].stdin.saved, bytes([7]) * 4)
        self.assertTrue(FakeEncoder.instances[0].stdin.closed)
        self.assertEqual(self.output.read_bytes(), b'encoded')

    def test_frame_gap_holds_previous_image(self):
        FakeBag.messages = [('/camera', raw_frame(1), stamp(0)), ('/camera', raw_frame(9), stamp(1))]
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            self.writer(fps=2).addBag(self.bag)
        self.assertEqual(FakeEncoder.instances[0].stdin.saved, bytes([1]) * 8 + bytes([9]) * 4)

    def test_encoding_failure_preserves_output(self):
        self.output.write_bytes(b'original')
        FakeBag.messages = [('/camera', raw_frame(), stamp(0))]
        FakeEncoder.code = 1
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg failed'):
                self.writer().addBag(self.bag)
        self.assertEqual(self.output.read_bytes(), b'original')
        self.assertEqual(set(self.root.iterdir()), {self.bag, self.output})

    def test_empty_topic_preserves_output(self):
        self.output.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, 'No supported image'):
            self.writer().addBag(self.bag)
        self.assertEqual(self.output.read_bytes(), b'original')

    def test_multiple_topics_get_distinct_files(self):
        FakeBag.messages = [('/left', raw_frame(1), stamp(0)), ('/right', raw_frame(2), stamp(0))]
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            video.RosVideoWriter().addBag(self.bag)
        self.assertTrue((self.root / 'sample_left.mp4').is_file())
        self.assertTrue((self.root / 'sample_right.mp4').is_file())

    def test_topic_filename_collision_is_detected(self):
        FakeBag.messages = [('/a/b', raw_frame(), stamp(0)), ('/a_b', raw_frame(), stamp(0))]
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            with self.assertRaisesRegex(ValueError, 'same output'):
                video.RosVideoWriter().addBag(self.bag)
        self.assertFalse((self.root / 'sample_a_b.mp4').exists())

    def test_two_bags_use_their_own_filename(self):
        second = self.root / 'second.bag'
        second.write_bytes(b'bag')
        FakeBag.messages = [('/camera', raw_frame(), stamp(0))]
        writer = video.RosVideoWriter()
        with patch.object(video.subprocess, 'Popen', FakeEncoder):
            writer.addBag(self.bag)
            writer.addBag(second)
        self.assertTrue((self.root / 'sample_camera.mp4').is_file())
        self.assertTrue((self.root / 'second_camera.mp4').is_file())

    def test_padded_rows_and_nonstandard_yuv_dimensions(self):
        msg = raw_frame()
        msg.data, msg.step = b'abXXcdXX', 4
        packed, fmt = video.prepare_image(msg)
        self.assertEqual((packed.data, fmt), (b'abcd', 'gray'))
        msg.encoding, msg.width, msg.height, msg.step = 'yuv422', 4, 1, 8
        packed, fmt = video.prepare_image(msg)
        self.assertEqual((packed.width, packed.height, fmt), (4, 1, 'uyvy422'))

    def test_big_endian_depth(self):
        msg = raw_frame(width=2, height=1)
        msg.encoding, msg.is_bigendian, msg.step, msg.data = '16UC1', True, 4, b'\x00\x01\x00\x02'
        self.assertEqual(video.prepare_image(msg)[1], 'gray16be')

    def test_png_compressed_image(self):
        import cv2
        ok, encoded = cv2.imencode('.png', np.full((2, 3, 3), 50, dtype=np.uint8))
        self.assertTrue(ok)
        frame, fmt = video.prepare_image(SimpleNamespace(format='bgr8; png compressed', data=encoded.tobytes()))
        self.assertEqual((frame.width, frame.height, fmt), (3, 2, 'bgr24'))
        self.assertEqual(frame.data, bytes([50]) * 18)

    def test_invalid_options_and_fractional_timestamps(self):
        for args in [['--fps', 'nan'], ['--rate', '0'], ['--start', '2', '--end', '1'], ['-o', 'out.mp4']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    video.RosVideoWriter().parseArgs(args + ['input.bag'])
        writer = video.RosVideoWriter()
        writer.parseArgs(['--start', '1.25', '--end', '2.5', 'input.bag'])
        self.assertEqual((writer.opt_start, writer.opt_end), (1.25, 2.5))

    def test_real_ffmpeg_one_frame_video(self):
        import cv2
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest('Install imageio-ffmpeg to run real video encoding')
        FakeBag.messages = [('/camera', raw_frame(100, width=16, height=16), stamp(0))]
        with patch.object(video, 'VIDEO_CONVERTER_TO_USE', imageio_ffmpeg.get_ffmpeg_exe()):
            self.writer().addBag(self.bag)
        capture = cv2.VideoCapture(str(self.output))
        try:
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (16, 16))
            self.assertFalse(capture.read()[0])
        finally:
            capture.release()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Export ROS image topics to fixed-frame-rate videos with ffmpeg.

Based on the original rosbag2video script by Maximilian Laiacker and Abel Gabor.
"""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from file_utils import atomic_output, validate_paths

MJPEG_VIDEO = 1
RAWIMAGE_VIDEO = 2
VIDEO_CONVERTER_TO_USE = 'ffmpeg'


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bags', nargs='+')
    parser.add_argument('--fps', type=positive_float, default=25.0)
    parser.add_argument('-r', '--rate', type=positive_float, default=1.0)
    parser.add_argument('-o', '--ofile', default='', help='explicit output; requires one bag and --topic')
    parser.add_argument('-t', '--topic', default='')
    parser.add_argument('-p', '--prefix', help='output filename prefix (one bag only)')
    parser.add_argument('-s', action='store_true', help='display images while converting')
    parser.add_argument('-v', action='store_true', help='show output filenames')
    parser.add_argument('--start', type=nonnegative_float, help='start ROS timestamp in seconds')
    parser.add_argument('--end', type=nonnegative_float, help='end ROS timestamp in seconds')
    return parser


def print_help():
    make_parser().print_help()


def prepare_image(msg):
    """Return packed image bytes and an ffmpeg pixel format; remove ROS row padding."""
    if hasattr(msg, 'format'):
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('Unable to decode compressed image')
        if image.ndim == 2:
            if image.dtype == np.uint8:
                pix_fmt = 'gray'
            elif image.dtype == np.uint16:
                image = image.astype('<u2', copy=False)
                pix_fmt = 'gray16le'
            else:
                raise ValueError('Unsupported compressed image depth')
        elif image.dtype == np.uint8 and image.shape[2] in (3, 4):
            pix_fmt = 'bgr24' if image.shape[2] == 3 else 'bgra'
        else:
            raise ValueError('Unsupported compressed image format')
        height, width = image.shape[:2]
        return SimpleNamespace(data=image.tobytes(), width=width, height=height), pix_fmt

    formats = {
        'mono8': ('gray', 1), '8UC1': ('gray', 1),
        'bgr8': ('bgr24', 3), 'rgb8': ('rgb24', 3),
        'bgra8': ('bgra', 4), 'rgba8': ('rgba', 4),
        'bayer_bggr8': ('bayer_bggr8', 1), 'bayer_rggb8': ('bayer_rggb8', 1),
        'bggr8': ('bayer_bggr8', 1), 'rggb8': ('bayer_rggb8', 1),
        'mono16': ('gray16be' if msg.is_bigendian else 'gray16le', 2),
        '16UC1': ('gray16be' if msg.is_bigendian else 'gray16le', 2),
        'yuv422': ('uyvy422', 2),
    }
    if msg.encoding not in formats:
        raise ValueError(f'Unsupported raw image encoding: {msg.encoding}')
    pix_fmt, bytes_per_pixel = formats[msg.encoding]
    width, height = msg.width, msg.height
    row_bytes = width * bytes_per_pixel
    if width <= 0 or height <= 0 or msg.step < row_bytes or len(msg.data) != msg.step * height:
        raise ValueError('Invalid image dimensions, step, or data length')
    if msg.encoding == 'yuv422' and width % 2:
        raise ValueError('yuv422 requires an even image width')
    data = bytes(msg.data)
    if msg.step != row_bytes:
        data = b''.join(data[row * msg.step:row * msg.step + row_bytes] for row in range(height))
    return SimpleNamespace(data=data, width=width, height=height), pix_fmt


class RosVideoWriter:
    def __init__(self, fps=25.0, rate=1.0, topic='', output_filename='',
                 display=False, verbose=False, start=None, end=None):
        self.fps, self.rate = positive_float(fps), positive_float(rate)
        self.opt_topic, self.opt_out_file = topic, output_filename
        self.opt_display_images, self.opt_verbose = display, verbose
        self.opt_start, self.opt_end = start, end
        self.opt_prefix = None
        self._used_outputs = set()
        self._reset_streams()

    def _reset_streams(self):
        self.t_first, self.t_video, self.t_file = {}, {}, {}
        self.p_avconv, self._frames, self._formats = {}, {}, {}

    def parseArgs(self, args):
        parser = make_parser()
        options = parser.parse_args(args)
        if options.start is not None and options.end is not None and options.end <= options.start:
            parser.error('--end must be greater than --start')
        if options.ofile and (not options.topic or len(options.bags) != 1):
            parser.error('--ofile requires --topic and exactly one bag')
        if options.prefix and len(options.bags) != 1:
            parser.error('--prefix requires exactly one bag')
        self.fps, self.rate = options.fps, options.rate
        self.opt_out_file, self.opt_topic = options.ofile, options.topic
        self.opt_prefix = options.prefix
        self.opt_start, self.opt_end = options.start, options.end
        self.opt_display_images, self.opt_verbose = options.s, options.v
        return options.bags

    def filter_image_msgs(self, topic, datatype, md5sum, msg_def, header):
        return (datatype in ('sensor_msgs/Image', 'sensor_msgs/CompressedImage') and
                (not self.opt_topic or topic == self.opt_topic))

    def _start_encoder(self, topic, msg, pix_fmt):
        prefix = self.opt_prefix if self.opt_prefix is not None else str(Path(self._bagfile).with_suffix(''))
        destination = Path(self.opt_out_file or (prefix + topic.replace('/', '_') + '.mp4'))
        validate_paths(self._bagfile, destination)
        key = destination.resolve()
        if key in self._used_outputs:
            raise ValueError(f'Multiple streams would overwrite the same output: {destination}')
        self._used_outputs.add(key)
        temporary = self._outputs.enter_context(atomic_output(destination))
        command = [VIDEO_CONVERTER_TO_USE, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-f', 'rawvideo', '-s', f'{msg.width}x{msg.height}', '-pix_fmt', pix_fmt,
                   '-r', str(self.fps), '-i', '-', '-an', str(temporary)]
        if self.opt_verbose:
            print(f'Writing {topic} to {destination}')
        self.p_avconv[topic] = subprocess.Popen(command, stdin=subprocess.PIPE)
        self._formats[topic] = (msg.width, msg.height, pix_fmt)

    def write_output_video(self, msg, topic, t, video_fmt, pix_fmt=''):
        if video_fmt != RAWIMAGE_VIDEO:
            raise ValueError('Images must be decoded before writing raw video')
        if not msg.data:
            raise ValueError(f'Empty image on {topic}')
        timestamp = t.to_sec()
        if not math.isfinite(timestamp):
            raise ValueError('Image timestamp must be finite')
        if topic not in self.t_first:
            self._start_encoder(topic, msg, pix_fmt)
            self.t_first[topic], self.t_file[topic] = timestamp, 0.0
            self.p_avconv[topic].stdin.write(msg.data)
            self.t_video[topic] = 1  # Next output frame index; frame zero is written immediately.
        else:
            if self._formats[topic] != (msg.width, msg.height, pix_fmt):
                raise ValueError(f'Image format changed on {topic}')
            offset = (timestamp - self.t_first[topic]) / self.rate
            if offset < self.t_file[topic]:
                raise ValueError(f'Image timestamps went backwards on {topic}')
            # Hold the preceding image between samples, rather than moving future frames backwards.
            while self.t_video[topic] / self.fps < offset - 1e-9:
                self.p_avconv[topic].stdin.write(self._frames[topic])
                self.t_video[topic] += 1
            if abs(self.t_video[topic] / self.fps - offset) <= 1e-9:
                self.p_avconv[topic].stdin.write(msg.data)
                self.t_video[topic] += 1
            self.t_file[topic] = offset
        self._frames[topic] = msg.data

    def _close_encoders(self):
        errors = []
        for topic, process in self.p_avconv.items():
            try:
                try:
                    process.stdin.close()
                finally:
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise RuntimeError(f'ffmpeg did not finish for {topic}')
                if code:
                    raise RuntimeError(f'ffmpeg failed for {topic} with exit code {code}')
            except (OSError, RuntimeError) as error:
                errors.append(str(error))
        self.p_avconv.clear()
        if errors:
            raise RuntimeError('; '.join(errors))

    def addBag(self, filename):
        if not Path(filename).is_file():
            raise FileNotFoundError(f'Input file does not exist: {filename}')
        if self.opt_out_file and not self.opt_topic:
            raise ValueError('An explicit output filename requires a single --topic')
        import rosbag
        self._bagfile = filename
        self._reset_streams()
        read_options = {'connection_filter': self.filter_image_msgs}
        if self.opt_start is not None or self.opt_end is not None:
            import rospy
            for key, value in [('start_time', self.opt_start), ('end_time', self.opt_end)]:
                if value is not None:
                    read_options[key] = value if hasattr(value, 'to_sec') else rospy.Time.from_sec(value)
        count = 0
        with ExitStack() as self._outputs:
            with rosbag.Bag(str(filename), 'r') as bag:
                try:
                    for topic, msg, timestamp in bag.read_messages(**read_options):
                        frame, pix_fmt = prepare_image(msg)
                        self.write_output_video(frame, topic, timestamp, RAWIMAGE_VIDEO, pix_fmt)
                        count += 1
                        if self.opt_display_images:
                            import cv2
                            from cv_bridge import CvBridge
                            bridge = CvBridge()
                            image = (bridge.compressed_imgmsg_to_cv2(msg, 'bgr8') if hasattr(msg, 'format')
                                     else bridge.imgmsg_to_cv2(msg, 'bgr8'))
                            cv2.imshow(topic, image)
                            if cv2.waitKey(1) & 0xFF == 27:
                                raise RuntimeError('Video export cancelled')
                    if count == 0:
                        raise ValueError(f'No supported image messages found in {filename}')
                finally:
                    try:
                        self._close_encoders()
                    finally:
                        if self.opt_display_images:
                            import cv2
                            cv2.destroyAllWindows()
        return count


def main(argv=None):
    import sys
    writer = RosVideoWriter()
    bags = writer.parseArgs(sys.argv[1:] if argv is None else argv)
    # Check every input before starting any output.
    for filename in bags:
        if not Path(filename).is_file():
            raise FileNotFoundError(f'Input file does not exist: {filename}')
        for other in bags:
            if writer.opt_out_file and Path(writer.opt_out_file).resolve() == Path(other).resolve():
                raise ValueError('Output must not overwrite an input bag')
    for filename in bags:
        writer.addBag(filename)
    print('Video export finished')


if __name__ == '__main__':
    main()

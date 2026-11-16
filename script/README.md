# Dataset processing scripts

Run these commands from the repository root in a Python 3 environment. Bag conversion requires ROS 1 (`rosbag` / `rostopic`) and the message packages used by the recording. Indoor processing needs NumPy, pandas, and SciPy; filtering also needs pykalman. Optical flow requires OpenCV and `cv_bridge`. Video export requires `ffmpeg` on `PATH`; compressed images also require NumPy and OpenCV, and `-s` requires `cv_bridge`.

All six entry points support `--help` without importing ROS or numerical packages.

## Commands

```bash
# Odometry to TUM (the topic must contain nav_msgs/Odometry messages)
python3 script/bag2tum.py -b input.bag -o trajectory.tum -t /firefly_sbx/vio/odom

# Indoor UWB to TUM, with the existing TOA calculation
python3 script/indoor_bag2gt.py -b indoor.bag -o indoor.tum -p /nlink_linktrack_tagframe0

# Outdoor RTK and IMU to TUM, with the existing 5*i sample association
python3 script/outdoor_bag2gt.py -b outdoor.bag -o outdoor.tum -p1 /ublox_driver/receiver_pvt -p2 /mavros/imu/data

# Existing Kalman filtering (the output updates x/y and preserves z/orientation)
python3 script/kalman_fillter.py -i trajectory.tum -o filtered.tum

# Optical-flow magnitude
python3 script/cal_optical_flow.py -b input.bag -t /camera/infra1/image_rect_raw --frame-skip 10

# Export one image topic, including a single-image recording
python3 script/rosbag2video.py --fps 25 -t /camera/color/image_raw -o flight.mp4 input.bag

# Export all supported image topics to separate files for each bag
python3 script/rosbag2video.py first.bag second.bag
```

Quote paths containing spaces. If omitted, trajectory output names are derived from the input stem: `bag.bag` becomes `bag.tum`, or `bag_toa_fit.tum` for the indoor converter. Filtering defaults to `<stem>_filtered.tum`.

Trajectory and video outputs replace existing files only after conversion succeeds. A failed export preserves the previous output and removes the temporary output. Input and output cannot refer to the same file. ROS CSV exports use separate temporary directories and do not invoke a shell.

Video `--start` and `--end` are absolute ROS timestamps in seconds and accept decimals. `--rate 2` produces double-speed playback. `-o` requires exactly one bag and an explicit `--topic`; `--prefix` also accepts one bag only. Raw image row padding is removed before encoding. Unsupported image encodings, backwards timestamps, and mid-stream format changes produce errors. The encoder is closed and its exit status is checked before publishing a video.

## Ground-truth limitations retained in this change

The indoor anchor coordinates remain the original unit-square placeholders. The outdoor converter still associates IMU row `5*i` with RTK row `i`; this is not timestamp synchronization. It now reports insufficient IMU samples before replacing the output. The coordinate conversion, TOA fitting, quaternion ordering, and filtering calculations have not been redesigned.

## Regression checks

The tests use synthetic data and mocked ROS bag readers; they do not replace validation with real ROS recordings. Run them in an isolated Python 3 environment with NumPy, pandas, SciPy, pykalman, and OpenCV installed:

```bash
python -m unittest discover -s tests -v
```

Installing `imageio-ffmpeg` enables the real single-frame video encoding test; otherwise that one test is skipped. The tests cover file preservation, literal subprocess arguments, repeat conversions, sample-count validation, unchanged numerical results, CLI parsing, pixel packing, multiple image topics/bags, encoder failure, and single-frame export.

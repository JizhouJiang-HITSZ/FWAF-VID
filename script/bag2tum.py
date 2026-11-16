"""Convert a ROS odometry topic to a TUM trajectory."""

import argparse
from pathlib import Path

from file_utils import atomic_output, validate_paths, write_tum_row


def convert_bag_to_tum(input_bag, output_file, topic='/firefly_sbx/vio/odom'):
    validate_paths(input_bag, output_file)
    import rosbag

    count = 0
    with rosbag.Bag(str(input_bag), 'r') as bag, atomic_output(output_file) as temporary:
        with temporary.open('w', encoding='utf-8') as output:
            for _, msg, _ in bag.read_messages(topics=[topic]):
                timestamp = msg.header.stamp.to_nsec() * 1e-9
                position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
                write_tum_row(output, [timestamp, position.x, position.y, position.z,
                                      orientation.x, orientation.y, orientation.z, orientation.w])
                count += 1
        if count == 0:
            raise ValueError(f'No odometry messages found on {topic}')
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--bag', required=True)
    parser.add_argument('-o', '--output-file', '--output_file')
    parser.add_argument('-p', '-t', '--topic', default='/firefly_sbx/vio/odom')
    args = parser.parse_args(argv)
    output = args.output_file or str(Path(args.bag).with_suffix('.tum'))
    count = convert_bag_to_tum(args.bag, output, args.topic)
    print(f'Conversion complete: {count} poses saved to {output}')


if __name__ == '__main__':
    main()

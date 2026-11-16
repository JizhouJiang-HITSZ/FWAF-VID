# coding:utf-8
"""Export indoor ground truth using the existing TOA calculation."""

import argparse
from pathlib import Path
import tempfile

from file_utils import atomic_output, export_topic_csv, require_columns, validate_paths, write_tum_row


def toa_residuals(params, anchor_positions, distances):
    import numpy as np
    x, y = params
    return np.sqrt((x - anchor_positions[:, 0]) ** 2 + (y - anchor_positions[:, 1]) ** 2) - distances


def fit_toa(anchor_positions, distances):
    from scipy.optimize import least_squares
    initial_guess = [0, 0]  # Initial guess for x, y
    result = least_squares(toa_residuals, initial_guess, args=(anchor_positions, distances))
    return result.x


def convert_to_tum_with_toa(bagfile, outfile, topic):
    validate_paths(bagfile, outfile)
    import numpy as np
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix='fwaf-indoor-') as directory:
        csv_file = Path(directory) / 'uwb.csv'
        export_topic_csv(bagfile, topic, csv_file)
        df = pd.read_csv(csv_file)
    require_columns(df, ['%time', 'field.pos_3d2'] +
                    [f'field.dis_arr{i}' for i in range(4)] +
                    [f'field.quaternion{i}' for i in range(4)], topic)

    # Existing placeholder geometry is unchanged; actual anchor positions are required.
    anchor_positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    with atomic_output(outfile) as temporary:
        with temporary.open('w', encoding='utf-8') as output:
            for i in range(len(df)):
                distances = np.array([df[f'field.dis_arr{j}'][i] for j in range(4)])
                pos_2d = fit_toa(anchor_positions, distances)
                result = [df['%time'][i] * 1e-9, pos_2d[0], pos_2d[1],
                          df['field.pos_3d2'][i]]
                result.extend(df[f'field.quaternion{j}'][i] for j in range(4))
                write_tum_row(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--bag', required=True, help='rosbag with UWB information')
    parser.add_argument('-o', '--output-file', '--output_file', help='output TUM file')
    parser.add_argument('-p', '--topic', '--topic_name', default='/nlink_linktrack_tagframe0')
    args = parser.parse_args(argv)
    output = args.output_file or str(Path(args.bag).with_name(Path(args.bag).stem + '_toa_fit.tum'))
    convert_to_tum_with_toa(args.bag, output, args.topic)
    print(f'Conversion finished. TUM data saved to {output}')


if __name__ == '__main__':
    main()

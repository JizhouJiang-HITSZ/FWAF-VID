# coding:utf-8
"""Export RTK position and IMU quaternion using the existing ground-truth calculation."""

import argparse
import math
from pathlib import Path
import tempfile

from file_utils import atomic_output, export_topic_csv, require_columns, validate_paths, write_tum_row


def to_xyz_3(M_lat, M_lon, M_alt, O_lat, O_lon, O_alt):
    Ea = 6378137  # Equatorial radius
    Eb = 6356752  # Polar radius
    M_lat = math.radians(M_lat)
    M_lon = math.radians(M_lon)
    O_lat = math.radians(O_lat)
    O_lon = math.radians(O_lon)
    Ec = Ea * (1 - (Ea - Eb) / Ea * ((math.sin(M_lat)) ** 2)) + M_alt
    Ed = Ec * math.cos(M_lat)
    d_lat = M_lat - O_lat
    d_lon = M_lon - O_lon
    x = d_lat * Ec
    y = d_lon * Ed
    z = M_alt - O_alt
    return x, y, z


def convert_to_tum(bagfile, outfile, rtk_topic, imu_topic):
    validate_paths(bagfile, outfile)
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix='fwaf-outdoor-') as directory:
        rtk_csv, imu_csv = Path(directory) / 'rtk.csv', Path(directory) / 'imu.csv'
        export_topic_csv(bagfile, rtk_topic, rtk_csv)
        export_topic_csv(bagfile, imu_topic, imu_csv)
        df1, df2 = pd.read_csv(rtk_csv), pd.read_csv(imu_csv)
    require_columns(df1, ['%time', 'field.latitude', 'field.longitude', 'field.altitude'], rtk_topic)
    require_columns(df2, [f'field.orientation.{axis}' for axis in 'xyzw'], imu_topic)
    # Preserve the original 5*i association. Timestamp alignment is a separate change.
    if 5 * (len(df1) - 1) >= len(df2):
        raise ValueError('Not enough IMU samples for the existing 5*i association; output was not changed')
    O_lat = df1['field.latitude'][0]
    O_lon = df1['field.longitude'][0]
    O_alt = df1['field.altitude'][0]

    with atomic_output(outfile) as temporary:
        with temporary.open('w', encoding='utf-8') as output:
            for i in range(len(df1)):
                x, y, z = to_xyz_3(df1['field.latitude'][i], df1['field.longitude'][i],
                                   df1['field.altitude'][i], O_lat, O_lon, O_alt)
                result = [df1['%time'][i] * 1e-9, x, y, z]
                result.extend(df2[f'field.orientation.{axis}'][5 * i] for axis in 'xyzw')
                write_tum_row(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--bag', required=True, help='rosbag with RTK and IMU information')
    parser.add_argument('-o', '--output-file', '--output_file', help='output TUM file')
    parser.add_argument('-p1', '--rtk-topic', default='/ublox_driver/receiver_pvt')
    parser.add_argument('-p2', '--imu-topic', default='/mavros/imu/data')
    args = parser.parse_args(argv)
    output = args.output_file or str(Path(args.bag).with_suffix('.tum'))
    convert_to_tum(args.bag, output, args.rtk_topic, args.imu_topic)
    print(f'Conversion finished. TUM data saved to {output}')


if __name__ == '__main__':
    main()

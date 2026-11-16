import argparse
from pathlib import Path

from file_utils import atomic_output, validate_paths

def kalman_filter(data, transition_covariance, observation_covariance):
    from pykalman import KalmanFilter
    # Initialize Kalman filter
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1, transition_covariance=transition_covariance, observation_covariance=observation_covariance)
    
    # Apply Kalman filter to smooth the data
    smoothed_data, _ = kf.filter(data)

    return smoothed_data.flatten()

def write_tum_data(path, original_data, smoothed_data):
    with open(path, 'w') as f:
        for i in range(len(original_data)):
            f.write(f"{original_data[i][0]:.10f} ")
            f.write(" ".join([f"{val:.10f}" for val in smoothed_data[i][:2]]))
            f.write(f" {original_data[i][3]:.10f} ")  # Add a space between the fourth and fifth columns
            f.write(" ".join([f"{val:.10f}" for val in original_data[i][4:]]))
            f.write("\n")
        f.close()

def main(input_file, output_file):
    validate_paths(input_file, output_file)
    import numpy as np
    import pandas as pd
    df = pd.read_csv(input_file, header=None, sep=r'\s+', comment='#')
    if df.empty or df.shape[1] != 8 or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('Expected nonempty TUM data with eight finite numeric columns')

    # Extract the second, third, fourth, and fifth columns
    selected_columns = df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]

    # Apply Kalman filter to each column with adjusted covariance values
    transition_covariance = 0.5  # Adjust as needed
    observation_covariance = 30.0  # Adjust as needed
    smoothed_data = np.apply_along_axis(kalman_filter, axis=0, arr=selected_columns.values, transition_covariance=transition_covariance, observation_covariance=observation_covariance)

    # Save the smoothed results to the output file in TUM format
    with atomic_output(output_file) as temporary:
        write_tum_data(temporary, df.values, smoothed_data)
    print(f"Kalman filtering completed. Results saved to {output_file}")

def cli(argv=None):
    parser = argparse.ArgumentParser(description='Apply the existing Kalman filter to a TUM trajectory')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output')
    args = parser.parse_args(argv)
    output = args.output or str(Path(args.input).with_name(Path(args.input).stem + '_filtered.tum'))
    main(args.input, output)


if __name__ == '__main__':
    cli()

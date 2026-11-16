#!/usr/bin/env python

import argparse
from pathlib import Path

class OpticalFlowMagnitude:
    def __init__(self, bag_file, image_topic):
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.bag_file = bag_file
        self.image_topic = image_topic
        self.previous_frame = None
        self.max_magnitude = 0  # Initialize the maximum optical flow magnitude
        self.image_diagonal = None  # Store the image diagonal in pixels

    def process_bag(self, frame_skip=10):
        if not isinstance(frame_skip, int) or frame_skip <= 0:
            raise ValueError('frame_skip must be a positive integer')
        import rosbag
        with rosbag.Bag(self.bag_file, 'r') as bag:
            frame_count = 0
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                if frame_count % frame_skip == 0:
                    self.process_image(msg)
                frame_count += 1
            if frame_count == 0:
                raise ValueError(f'No images found on {self.image_topic}')

    def process_image(self, img_msg):
        import cv2
        import numpy as np
        # Convert ROS Image message to OpenCV image
        current_frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Get image dimensions and calculate diagonal in pixels
        if self.image_diagonal is None:
            height, width = gray_frame.shape
            self.image_diagonal = np.sqrt(width**2 + height**2)
            print(f"Image diagonal (in pixels): {self.image_diagonal}")

        if self.previous_frame is not None and self.previous_frame.shape == gray_frame.shape:
            # Calculate optical flow using the Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(self.previous_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = self.compute_flow_magnitude(flow)

            # Normalize the optical flow magnitude
            normalized_magnitude = magnitude / self.image_diagonal

            # Update the maximum normalized magnitude
            self.max_magnitude = max(self.max_magnitude, np.max(normalized_magnitude))
            print(f"Current Optical Flow Magnitude (normalized):\n{normalized_magnitude}")
            print(f"Max Optical Flow Magnitude (normalized) so far: {self.max_magnitude}")

        # # Calculate optical flow using the Lucas-Kanade method
        # p0 = cv2.goodFeaturesToTrack(self.previous_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
        # p1, st, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, gray_frame, p0, None)
        # if p1 is not None:
        #     flow = p1 - p0
        #     magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        #     normalized_magnitude = magnitude / self.image_diagonal
        #     self.max_magnitude = max(self.max_magnitude, np.max(normalized_magnitude))
        #     print(f"Max Optical Flow Magnitude (normalized) so far: {self.max_magnitude}")

        # Update the previous frame
        self.previous_frame = gray_frame

    def compute_flow_magnitude(self, flow):
        import numpy as np
        # Compute the magnitude of the optical flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude

def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure normalized optical-flow magnitude in a rosbag')
    parser.add_argument('-b', '--bag', required=True)
    parser.add_argument('-t', '--topic', required=True)
    parser.add_argument('--frame-skip', type=int, default=10)
    args = parser.parse_args(argv)
    if args.frame_skip <= 0:
        parser.error('--frame-skip must be positive')
    if not Path(args.bag).is_file():
        parser.error(f'Input file does not exist: {args.bag}')
    processor = OpticalFlowMagnitude(args.bag, args.topic)
    processor.process_bag(frame_skip=args.frame_skip)


if __name__ == '__main__':
    main()

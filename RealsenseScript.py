import cv2
import numpy as np
import time

# First import the library
import pyrealsense2 as rs

def main():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    pc = rs.pointcloud()
    pipeline.start()

    temporal_filter = rs.temporal_filter()
    colorizer = rs.colorizer(2)
    align = rs.align(rs.stream.color)

    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_rs = frames.get_color_frame()
        color = np.asanyarray(color_rs.get_data())
        cv2.VideoWriter()

    pipeline.stop()


if __name__ == '__main__':
    main()
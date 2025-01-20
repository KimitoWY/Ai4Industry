import cv2
import numpy as np
from matplotlib import pyplot as plt
from extract_frames import VideoProcessor
import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

if __name__ == "__main__":
    VideoProcessor.extract_frames("./data/output.mp4", "./output/")

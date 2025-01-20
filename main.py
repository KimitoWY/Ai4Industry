import cv2
import numpy as np
from matplotlib import pyplot as plt
from extract_frames import VideoProcessor
from image_edge_processor import ImageEdgeProcessor
import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

if __name__ == "__main__":
    VideoProcessor.extract_frames("./data/20240914_target.mp4", "./output/")
    ImageEdgeProcessor.process_images_from_folder('./output/', "./canny/", 1, 1600)

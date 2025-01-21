from extract_frames import VideoProcessor
from image_edge_processor import ImageEdgeProcessor
import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

if __name__ == "__main__":
    # VideoProcessor.extract_frames("./data/20240914_target.mp4", "./output/")
    # ImageEdgeProcessor.process_images_from_folder('./output/', "./canny/", 1, 1600)

    # images, edges = ImageEdgeProcessor.detect_edges('./imageTest/frame_9004.png', 4)
    # ImageEdgeProcessor.display_images(images, edges)

    # roi_corners = [[(50, 720), (640, 360), (1230, 720)]]
    # masked_edges = ImageEdgeProcessor.detect_and_mask_edges('./imageTest/frame_9004.png', './masked_frame_9004.png', roi_corners)
    # ImageEdgeProcessor.display_images(edges, masked_edges)

    # curves_image = ImageEdgeProcessor.extract_large_curves(edges, './curves_frame_9004.png')
    # ImageEdgeProcessor.display_images(edges, curves_image)

    ImageEdgeProcessor.new_process_video('./videoTest/test.mp4',4)

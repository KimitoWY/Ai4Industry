import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class ImageEdgeProcessor:

    @staticmethod
    def display_images(original, edges):
        """
        Display the original and edge-detected images side by side.

        Args:
            original (numpy.ndarray): The original grayscale image.
            edges (numpy.ndarray): The edge-detected image.
        """
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(original, cmap='gray')
        plt.title("Image d'entrée")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.title("Contours détectés")
        plt.axis('off')

        plt.show()

    @staticmethod
    def detect_edges(image_path, scale=2, interpolation=cv2.INTER_AREA):
        """
        Detect edges in an image.

        Args:
            image_path (str): Path to the input image.
            scale (int): Scaling factor to resize the image.
            interpolation: Interpolation method for resizing.

        Returns:
            tuple: The resized image and the edge-detected image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width = image.shape[1] // scale
        height = image.shape[0] // scale
        image = cv2.resize(image, (width, height), interpolation=interpolation)

        # print("taille de l'image: ", image.shape, "\n")

        edges = cv2.Canny(image, 150, 250)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return image, edges

    @staticmethod
    def process_images_from_folder(folder_path, output_folder,start_index, end_index, scale=5, interpolation=cv2.INTER_AREA):
        """
        Process and detect edges in a sequence of images from a folder.

        Args:
            folder_path (str): Path to the folder containing images.
            start_index (int): Starting frame index.
            end_index (int): Ending frame index.
            scale (int): Scaling factor to resize images.
            interpolation: Interpolation method for resizing.
        """
        os.makedirs(output_folder, exist_ok=True)
        for i in range(start_index, end_index + 1):
            image_path = os.path.join(folder_path, f'frame_{i:04d}.jpg')
            if os.path.exists(image_path):
                image, edges = ImageEdgeProcessor.detect_edges(image_path, scale, interpolation)
                cv2.imwrite(output_folder + f'frame_{i:04d}.jpg' , edges)
            else:
                # print(f"Image {image_path} not found.")
                pass

    @staticmethod
    def process_video(video_path, scale=2, interpolation=cv2.INTER_AREA):
        """
        Process and detect edges in a video, and save the output as video files.

        Args:
            video_path (str): Path to the input video.
            scale (int): Scaling factor to resize video frames.
            interpolation: Interpolation method for resizing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = min(25, input_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_original = cv2.VideoWriter('original_output.avi', fourcc, fps, (width, height), False)
        out_edges = cv2.VideoWriter('edges_output.avi', fourcc, fps, (width, height), False)

        delay = int(1000 / fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (width, height), interpolation=interpolation)

            edges = cv2.Canny(gray_resized, 150, 380)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            out_original.write(gray_resized)
            out_edges.write(edges)

            combined = np.hstack((gray_resized, edges))
            cv2.imshow('Original and Edges', combined)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        out_original.release()
        out_edges.release()
        cv2.destroyAllWindows()

# Example usage
# ImageEdgeProcessor.process_images_from_folder('../imageTest/', 1, 2, 8)
# ImageEdgeProcessor.process_video('../videoTest/test.mp4', 4)

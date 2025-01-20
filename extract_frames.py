import cv2
import os

class VideoProcessor:

    @staticmethod
    def extract_frames(video_path, output_folder, consecutive_frames=4, frame_to_skip=12):
        """
        Extracts frames from a video file and saves them as image files.
        Args:
        video_path (str): The path to the video file to process.
        output_folder (str): The directory where extracted frames will be saved.
        consecutive_frames (int): The number of consecutive frames to extract.
        frame_to_skip (int): The number of frames to skip between each consecutive frame serie
        Raises:
        FileNotFoundError: If the video file cannot be opened.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return


        # skip the first frame_to_skip frames
        VideoProcessor.skip_frames(cap, frame_to_skip)

        frame_count = frame_to_skip
        frame_extracted = 0
        frame_consecutively_read = 0
        while True:
            if(frame_consecutively_read < consecutive_frames):
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if no frames are left or there's an error

                # Save the current frame as an image file
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_consecutively_read += 1
                frame_extracted += 1
                frame_count += 1
            else :
                VideoProcessor.skip_frames(cap, frame_to_skip)
                frame_consecutively_read = 0
                frame_count += frame_to_skip

        cap.release()
        print(f"Extracted {frame_extracted} frames to '{output_folder}'")

    @staticmethod
    def skip_frames(cap, frame_to_skip):
        for i in range(frame_to_skip):
            ret, frame = cap.read()
            if not ret:
                # print("Error: Unable to read video file")
                return

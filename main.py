from extract_frames import VideoProcessor
from image_edge_processor import ImageEdgeProcessor
from model import create_unet_model, preprocess_images
from tensorflow.keras.optimizers import Adam
from PIL import Image
from write_csv import generate_csv_from_directory
import os
from codecarbon import EmissionsTracker
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

def PosToXY(lat,lon):

    # Ouvrir l'image
    image = Image.open('data/Annecy.png')

    # Obtenir les dimensions de l'image
    width, height = image.size

    xratio = (lon - 0.211025) / (0.2136273 - 0.211025)
    yratio = 1.0 - ((lat - 47.9411527) / (47.943917 - 47.9411527))
    x = int(float(width - 512)*xratio+0.5) + (512 / 2) + 64
    y = int(float(height - 512)*yratio+0.5) + (512 / 2) + 64
    return (x,y)

if __name__ == "__main__":
    directory_path = "output"  # Replace with the path to your directory
    output_csv_path = "labels.csv"  # Replace with the desired output CSV file path
    generate_csv_from_directory(directory_path, output_csv_path)
    # VideoProcessor.extract_frames("./data/20240914_target.mp4", "./output/")
    # ImageEdgeProcessor.process_images_from_folder('./output/', "./canny/", 1, 1600)
    #tracker = EmissionsTracker()
    #tracker.start()


    # model = create_unet_model(input_shape=(256, 256, 1))
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # # Train the model
    # history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, batch_size=8)

    # Step 3: Load the training data
    # image_dir = "output"  # Directory containing processed edge-detected images
    # label_dir = "canny"  # Directory containing manually annotated labels (masks)
    # img_size = (256, 256)

    # train_images, train_labels = preprocess_images(image_dir, label_dir, img_size)

    # # Step 4: Compile the model
    # model = create_unet_model(input_shape=(256, 256, 1))
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # # Step 5: Train the model
    # history = model.fit(
    #     train_images, train_labels,
    #     validation_split=0.2,
    #     epochs=5,
    #     batch_size=8
    # )

    # # Step 6: Save the model
    # model.save("track_reconstruction_model.h5")

    # # images, edges = ImageEdgeProcessor.detect_edges('./imageTest/frame_9004.png', 4)
    # # ImageEdgeProcessor.display_images(images, edges)

    # # roi_corners = [[(50, 720), (640, 360), (1230, 720)]]
    # # masked_edges = ImageEdgeProcessor.detect_and_mask_edges('./imageTest/frame_9004.png', './masked_frame_9004.png', roi_corners)
    # # ImageEdgeProcessor.display_images(edges, masked_edges)

    # # curves_image = ImageEdgeProcessor.extract_large_curves(edges, './curves_frame_9004.png')
    # # ImageEdgeProcessor.display_images(edges, curves_image)

    ImageEdgeProcessor.new_process_video('./videoTest/test.mp4',4)
    #emissions : float = tracker.stop()
    #print(emissions)

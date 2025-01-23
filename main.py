from src.extract_frames import VideoProcessor
from src.image_edge_processor import ImageEdgeProcessor
from src.model import load_data, create_classification_model
from extract_frames import VideoProcessor
from image_edge_processor import ImageEdgeProcessor
from map_satellite import generate_map
from model import load_data, create_classification_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from src.write_csv import generate_csv_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import load_model
from src.model import predict_kart_position
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
    # directory_path = "output"  # Replace with the path to your directory
    # output_csv_path = "labels.csv"  # Replace with the desired output CSV file path
    # generate_csv_from_directory(directory_path, output_csv_path)
    # VideoProcessor.extract_frames("./data/20240914_target.mp4", "./output/")
    # ImageEdgeProcessor.process_images_from_folder('./output/', "./canny/", 1, 1600)
    tracker = EmissionsTracker()
    tracker.start()


    # Step 3: Load the training data
    # image_dir = "output"  # Directory containing processed edge-detected images
    # csv_file = "labels copy.csv"  # Directory containing manually annotated labels (masks)
    # img_size = (256, 256)

    # images, labels = load_data(image_dir, csv_file, img_size)

    # # Reshape images to add channel dimension
    # images = images.reshape(-1, img_size[0], img_size[1], 1)

    # # Convert labels to categorical format
    # num_classes = 4
    # labels_categorical = to_categorical(labels, num_classes=num_classes)

    # # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    # # Create and train the model
    # model = create_classification_model(input_shape=(img_size[0], img_size[1], 1), num_classes=num_classes)
    # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

    # # Save the trained model
    # model.save("kart_position_model.h5")

    # model = load_model("kart_position_model.h5")
    # image_path = "canny/frame_0591.jpg"
    # predicted_class = predict_kart_position(image_path, model)
    # print(f"Predicted class: {predicted_class}")

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

    # ImageEdgeProcessor.new_process_video('./videoTest/test.mp4',4)
    ImageEdgeProcessor.new_process_video('./data/20240914_target.mp4',4)
    emissions : float = tracker.stop()
    print("Emissions : ", emissions, "kg CO2 \n")
    
    # ImageEdgeProcessor.new_process_video('./data/20240914_target.mp4',4)
    # emissions : float = tracker.stop()
    # print(emissions)
    
    # Generate the map
    generate_map()

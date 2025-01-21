from extract_frames import VideoProcessor
from image_edge_processor import ImageEdgeProcessor
from model import create_unet_model, preprocess_images
from tensorflow.keras.optimizers import Adam
import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

if __name__ == "__main__":
    # VideoProcessor.extract_frames("./data/20240914_target.mp4", "./output/")
    # ImageEdgeProcessor.process_images_from_folder('./output/', "./canny/", 1, 1600)


    # model = create_unet_model(input_shape=(256, 256, 1))
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # # Train the model
    # history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, batch_size=8)

    # Step 3: Load the training data
    image_dir = "output"  # Directory containing processed edge-detected images
    label_dir = "canny"  # Directory containing manually annotated labels (masks)
    img_size = (256, 256)

    train_images, train_labels = preprocess_images(image_dir, label_dir, img_size)

    # Step 4: Compile the model
    model = create_unet_model(input_shape=(256, 256, 1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Step 5: Train the model
    history = model.fit(
        train_images, train_labels,
        validation_split=0.2,
        epochs=25,
        batch_size=8
    )

    # Step 6: Save the model
    model.save("track_reconstruction_model.h5")

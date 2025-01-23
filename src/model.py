import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
def load_data(image_dir, labels_path, img_size=(256, 256)):
    """
    Load images and their corresponding labels.

    Args:
        image_dir (str): Directory containing images.
        labels_path (str): Path to CSV file with image labels.
        img_size (tuple): Target size for resizing images.

    Returns:
        Tuple: (images, labels).
    """
    labels_df = pd.read_csv(labels_path)
    images = []
    labels = []
    #     image_files = sorted(os.listdir(image_dir))
    #     label_files = sorted(os.listdir(label_dir))

    for _, row in labels_df.iterrows():
        img_path = os.path.join(image_dir, row['name'])
        label = row['class']

        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, img_size)
            image = image / 255.0  # Normalize
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

# Build the classification model
def create_classification_model(input_shape, num_classes):
    """
    Create a CNN for classification.

    Args:
        input_shape (tuple): Shape of input images (H, W, C).
        num_classes (int): Number of output classes.

    Returns:
        Model: Compiled classification model.
    """
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def predict_kart_position(image_path, model, img_size=(256, 256)):
    """
    Predict the kart's position in an image.

    Args:
        image_path (str): Path to the input image.
        model: Trained classification model.
        img_size (tuple): Target size for resizing images.

    Returns:
        int: Predicted class (0: off-road, 1: left, 2: middle, 3: right).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize
    image = image.reshape(1, img_size[0], img_size[1], 1)

    prediction = model.predict(image)
    return np.argmax(prediction)

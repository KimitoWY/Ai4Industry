# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# import numpy as np
# import os

# def create_unet_model(input_shape=(256, 256, 1)):
#     inputs = Input(input_shape)

#     # Encoder
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     # Decoder
#     up1 = UpSampling2D(size=(2, 2))(pool2)
#     up1 = concatenate([up1, conv2])
#     conv3 = Conv2D(128, 3, activation='relu', padding='same')(up1)
#     conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

#     up2 = UpSampling2D(size=(2, 2))(conv3)
#     up2 = concatenate([up2, conv1])
#     conv4 = Conv2D(64, 3, activation='relu', padding='same')(up2)
#     conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)

#     outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv4)

#     model = Model(inputs, outputs)
#     return model

# def preprocess_images(image_dir, label_dir, img_size=(256, 256)):
#     """
#     Load and preprocess images and labels from directories.
#     Args:
#     image_dir (str): Path to the directory containing input images.
#     label_dir (str): Path to the directory containing label masks.
#     img_size (tuple): Target size for resizing images.
#     Returns:
#     Tuple of numpy arrays: (images, labels).
#     """
#     images = []
#     labels = []

#     image_files = sorted(os.listdir(image_dir))
#     label_files = sorted(os.listdir(label_dir))

#     for img_file, lbl_file in zip(image_files, label_files):
#         img_path = os.path.join(image_dir, img_file)
#         lbl_path = os.path.join(label_dir, lbl_file)
#         print(img_path)

#         # Load and preprocess the image
#         image = load_img(img_path, target_size=img_size, color_mode='grayscale')
#         image = img_to_array(image) / 255.0  # Normalize to [0, 1]

#         # Load and preprocess the label
#         label = load_img(lbl_path, target_size=img_size, color_mode='grayscale')
#         label = img_to_array(label) / 255.0  # Normalize to [0, 1]

#         images.append(image)
#         labels.append(label)

#     return np.array(images), np.array(labels)

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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

# Paths
# image_dir = "canny"  # Directory with images
# labels_path = "labels.csv"  # CSV file with image labels

# # Load data
# img_size = (256, 256)
# images, labels = load_data(image_dir, labels_path, img_size)

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

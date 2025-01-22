from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import numpy as np
import os

def create_unet_model(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(pool2)
    up1 = concatenate([up1, conv2])
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(up1)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

    up2 = UpSampling2D(size=(2, 2))(conv3)
    up2 = concatenate([up2, conv1])
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up2)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv4)

    model = Model(inputs, outputs)
    return model

def preprocess_images(image_dir, label_dir, img_size=(256, 256)):
    """
    Load and preprocess images and labels from directories.
    Args:
    image_dir (str): Path to the directory containing input images.
    label_dir (str): Path to the directory containing label masks.
    img_size (tuple): Target size for resizing images.
    Returns:
    Tuple of numpy arrays: (images, labels).
    """
    images = []
    labels = []

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    for img_file, lbl_file in zip(image_files, label_files):
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, lbl_file)
        print(img_path)

        # Load and preprocess the image
        image = load_img(img_path, target_size=img_size, color_mode='grayscale')
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]

        # Load and preprocess the label
        label = load_img(lbl_path, target_size=img_size, color_mode='grayscale')
        label = img_to_array(label) / 255.0  # Normalize to [0, 1]

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

#!/usr/bin/env python3
"""
  keras_image_classify.py
  Image classification from scratch

  Jeff Holmes
  06/30/2020

  Reference:
    https://keras.io/examples/vision/image_classification_from_scratch/
    https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.util import compat

CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH = [], [], []


def filter():
    """
    Filter out corrupted images
    """
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)


def show_data(train_ds):
    """
    Visualize the data
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()


def generate_data():
    """
    Generate Datasets

    @return:
    """
    image_size = (180, 180)
    batch_size = 32

    # The image_dataset_from_directory function is not in version 2.2
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Save datasets to file
    # np.savez('cats_dogs.npz', train_ds=train_ds, val_ds=val_ds)

    return train_ds, val_ds


def generate_data_flow():
    """
    Load using keras.preprocessing
    A simple way to load images is to use tf.keras.preprocessing.

    @return:
    """
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    # STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    data_dir = pathlib.Path("./PetImages")

    # The directory contains 5 sub-directories, one per class:
    image_count = len(list(data_dir.glob("*/*.jpg")))
    print(f"image_count = {image_count}")

    CLASS_NAMES = np.array(
        [
            item.name
            for item in data_dir.glob("*")
            if item.name not in [".DS_Store", "LICENSE.txt"]
        ]
    )
    print(f"CLASS_NAMES = {CLASS_NAMES}")

    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_ds = image_generator.flow_from_directory(
        directory=str(data_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes=list(CLASS_NAMES),
    )
    val_ds = train_ds


def generate_data_tensor():
    """
    Write a short pure-tensorflow function that converts a file path to an (img, label) pair
    @return:
    """
    IMG_HEIGHT = 180
    IMG_WIDTH = 180

    data_dir = pathlib.Path("./PetImages")

    # Load using keras.preprocessing
    # A simple way to load images is to use tf.keras.preprocessing.

    # The directory contains 5 sub-directories, one per class:
    image_count = len(list(data_dir.glob("*/*.jpg")))
    print(f"image_count = {image_count}")

    CLASS_NAMES = np.array(
        [
            item.name
            for item in data_dir.glob("*")
            if item.name not in [".DS_Store", "LICENSE.txt"]
        ]
    )
    print(f"CLASS_NAMES = {CLASS_NAMES}")

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"))

    for f in list_ds.take(5):
        print(f.numpy())

    # Use Dataset.map to create a dataset of image, label pairs.
    # Set num_parallel_calls so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=3)

    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    return labeled_ds


def augment_data(train_ds):
    """
    Using image data augmentation
    """
    data_augmentation = keras.Sequential(
        [
            layers.preprocessing.RandomFlip("horizontal"),
            layers.preprocessing.RandomRotation(0.1),
        ]
    )

    # We can visualize what the augmented samples look like by applying data_augmentation
    # repeatedly to the first image in the dataset.
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


def make_model(input_shape, num_classes):
    """
    Build a model

    @param input_shape:
    @param num_classes:
    @return:
    """
    inputs = keras.Input(shape=input_shape)

    # Perform data augmentation
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.preprocessing.RandomFlip("horizontal"),
    #         layers.preprocessing.RandomRotation(0.1),
    #     ]
    # )

    # augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Configure the dataset for performance
    # train_ds = train_ds.prefetch(buffer_size=32)
    # val_ds = val_ds.prefetch(buffer_size=32)

    # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.MaxPooling2D(pool_size=3, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters=size, kernel_size=1, strides=2, padding="same")(
            previous_block_activation
        )
        # x = layers.add([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(filters=1024, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)


def main():
    image_size = (180, 180)
    batch_size = 32
    input_size = image_size + (3,)

    # Load the data (raw data download)

    # Filter out corrupted images
    # filter()

    # Generate datasets
    # train_ds, val_ds = generate_data()
    generate_data_tensor()

    # Visualize the data
    # show_data(train_ds)

    # Build a model
    model = make_model(input_shape=input_size, num_classes=2)
    # keras.utils.plot_model(model, show_shapes=True)

    # Convert from NCHW to NHWC
    # train_nchw = tf.placeholder(tf.float32, [None, 3, 32, 32])  # input batch
    # out = tf.transpose(x, [0, 2, 3, 1])
    # print(out.get_shape())  # the shape of out is [None, 200, 300, 3]

    # flatten the leading dimensions
    # batch_shape = tf.shape(input0)[:-3]
    # input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    # input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # # NHWC to NCHW
    # input0 = tf.transpose(input0, [0, 3, 1, 2])
    # input1 = tf.transpose(input1, [0, 3, 1, 2])

    # Train the model
    # epochs = 50
    epochs = 2

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
    # ]
    #
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

    # Run inference on new data
    # Note that data augmentation and dropout are inactive at inference time.
    img = keras.preprocessing.image.load_img(
        "PetImages/Cat/6779.jpg", target_size=image_size
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )


if __name__ == "__main__":
    print()
    main()
    print("\nDone!")

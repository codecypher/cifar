#!/usr/bin/env python3
"""
  cifar10_90.py

  References:
    https://github.com/abhijeet3922/Object-recognition-CIFAR-10
    https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
"""
import numpy as np
import time
import math
import os
import sys
import argparse

from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.utils as np_utils

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


epochs = 20
batch_size = 64


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate


def show_images():
    """
    Show samples from each class
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_train.shape
    num_classes = len(np.unique(y_train))

    # If using tensorflow, set image dimensions order
    if K.backend() == "tensorflow":
        K.common.set_image_dim_ordering("th")

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    fig = plt.figure(figsize=(8, 3))

    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:] == i)[0]
        x_idx = X_train[idx, ::]
        img_num = np.random.randint(x_idx.shape[0])
        im = np.transpose(x_idx[img_num, ::], (1, 2, 0))
        ax.set_title(class_names[i])
        plt.imshow(im)

    plt.show()


def get_elapsed_time(start, end):
    """
    Compute elapsed time.

    @param start: start time
    @param end:   end time
    @return:      elapsed time (string)
    """
    diff = end - start
    days, hours, minutes = [0, 0, 0]
    s_time = []
    if diff > 86400:  # day
        days = math.floor(diff / 86400)
        diff = diff - days * 86400
    if diff > 3600:  # hour
        hours = math.floor(diff / 3600)
        diff = diff - hours * 3600
    if diff > 60:  # minute
        minutes = math.floor(diff / 60)
        diff = diff - minutes * 60

    if days > 0:
        s_time = "{0} days {1} hrs {2} min {3:.4f} sec".format(days, hours, minutes, diff)
        # print(f"{days} days {hours} hrs {minutes} min {diff:.4f} sec")
    elif hours > 0:
        s_time = "{0} hrs {1} min {2:.4f} sec".format(hours, minutes, diff)
        # print(f"{hours} hrs {minutes} min {diff:.4f} sec")
    elif minutes > 0:
        s_time = "{0} min {1:.4f} sec".format(minutes, diff)
        # print(f"{minutes} min {diff:.4f} sec")
    else:
        s_time = "{0:.4f} sec".format(diff)
        # print(f"{diff: .4f} sec")

    return s_time


def get_timestamp():
    """
    Compute timestamp

    @return:
    """
    # Calling now() function
    today = datetime.now()
    print("Current date and time is", today)

    s_timestamp = "{0}{1:02d}{02:02d}-{3:02d}{4:02d}{5:02d}".format(
        today.year, today.month, today.day, today.hour, today.minute, today.second
    )

    return s_timestamp


def accuracy(test_x, test_y, model):
    """
    Compute test accuracy

    @param test_x:
    @param test_y:
    @param model:
    @return:
    """
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]
    return accuracy * 100


def plot_model_history(model_history):
    """
    Plot model accuracy and loss

    @param model_history:
    @return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history for accuracy
    axs[0].plot(
        range(1, len(model_history.history["acc"]) + 1), model_history.history["acc"]
    )
    axs[0].plot(
        range(1, len(model_history.history["val_acc"]) + 1),
        model_history.history["val_acc"],
    )
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(
        np.arange(1, len(model_history.history["acc"]) + 1),
        len(model_history.history["acc"]) / 10,
    )
    axs[0].legend(["train", "val"], loc="best")

    # Summarize history for loss
    axs[1].plot(
        range(1, len(model_history.history["loss"]) + 1), model_history.history["loss"]
    )
    axs[1].plot(
        range(1, len(model_history.history["val_loss"]) + 1),
        model_history.history["val_loss"],
    )
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(
        np.arange(1, len(model_history.history["loss"]) + 1),
        len(model_history.history["loss"]) / 10,
    )
    axs[1].legend(["train", "val"], loc="best")

    file_name = "model-history.png"

    plt.savefig(file_name)  # Save plot to file
    # plt.show()            # Show plot
    plt.clf()  # Clear current figure
    plt.close(fig)


def preprocess():
    """
    Data preprocessing

    @return:
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_test.shape
    num_classes = len(np.unique(y_train))

    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Convert class labels to binary class labels
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    print(f"num_classes = {y_train.shape[1]}")

    print(X_train.shape[0], "Train samples")
    print(X_test.shape[0], "Test samples")

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Data Sanity Check
    print(f"\nnum_classes = {y_test.shape[1]}")
    print(f"X_test shape: {X_test.shape}")

    # Save datasets to file
    np.savez(
        "cifar10_80.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )


def create_model(name, num_classes, X_train, X_valid, y_train, y_valid):
    """
    Create model for 90% accuracy

    @return:
    """
    epochs = 10
    batch_size = 64
    rms_lr = 0.001
    rms_decay = 1e-6
    weight_decay = 1e-4

    # We expect our inputs to be RGB images of arbitrary size
    inputs = keras.Input(shape=(3, 32, 32))

    x = Sequential()(inputs)
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
        activation="elu",
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    # Finally, we add a classification layer.
    outputs = Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    # Training
    opt_rms = keras.optimizers.RMSprop(learning_rate=rms_lr, decay=rms_decay)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt_rms, metrics=["accuracy"]
    )

    start = time.time()
    model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=X_train.shape[0] // batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_valid, y_valid),
        callbacks=[LearningRateScheduler(lr_schedule)],
    )
    end = time.time()
    print(f"Model took [{get_elapsed_time(start, end)}] to train")

    return model


def evaluate_model():
    """
    Evaluate model

    @return:
    """
    batch_size = 128

    # Load data from file
    npzfile = np.load("cifar10_70.npz")
    print(npzfile.files)

    X_train = npzfile["X_train"]
    X_test = npzfile["X_test"]
    y_train = npzfile["y_train"]
    y_test = npzfile["y_test"]

    # load json and create model
    # json_file = open('models/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)

    # load weights into new model
    # model.load_weights("models/model.h5")
    # print("Loaded model from disk")

    # Return a compiled model (identical to the previous one)
    model = load_model("model_cnn_70.h5")

    # Training
    scores = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
    # print('\nTrain result: accuracy: %.3f loss: %.3f' % (scores[1], scores[0]))
    print(f"\nTrain result: accuracy: {scores[1]*100:.3f}%  loss: {scores[0]*100:.3f}%")

    # Testing
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    # print('\nTest result: accuracy: %.3f loss: %.3f' % (scores[1], scores[0]))
    print(f"\nTest result: accuracy: {scores[1]*100:.3f}%  loss: {scores[0]*100:.3f}%")


def main():
    # Get script filename instead of path
    file_name = os.path.basename(sys.argv[0])
    msg = sys.argv[0]
    msg = file_name

    # Initialize parser
    parser = argparse.ArgumentParser(description=msg)
    # parser = argparse.ArgumentParser()

    # Add optional arguments
    parser.add_argument(
        "-i", "--images", dest="images", action="store_true", help="Show images"
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Preprocess data",
    )
    parser.add_argument(
        "-c", "--create", dest="create", action="store_true", help="Create model"
    )
    parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true", help="Evaluate model"
    )

    # parser.add_argument("-p", "--preprocess", help="Preprocess data")
    # parser.add_argument("-v", "--verbose", dest="verbose",
    #                     action="store_true", help="verbose mode")

    # Read arguments from command line
    args = parser.parse_args()

    if args.images:
        show_images()
    elif args.preprocess:
        preprocess()
        # print(f"preprocess: {args.preprocess}")
    elif args.create:
        create_model()
    elif args.evaluate:
        evaluate_model()


if __name__ == "__main__":
    print()
    main()
    print("\nDone!")

#!/usr/bin/env python3
"""
  cifar10.py
  CNN for CIFAR-10 dataset

  Jeff Holmes
  06/30/2020

  References:
      https://keras.io/getting_started/intro_to_keras_for_engineers/
      https://keras.io/guides/training_with_built_in_methods/#api-overview-a-first-endtoend-example
      https://keras.io/api/
"""
import numpy as np
import time
import math
import os
import sys
import argparse
import matplotlib.pyplot as plt

from datetime import datetime
from pprint import pprint
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.utils as np_utils

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay, PolynomialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator


epochs = 200
batch_size = 128
# epochs = 20
# batch_size = 64


# def lr_schedule(epoch):
#     lrate = 0.001
#     if epoch > 75:
#         lrate = 0.0005
#     elif epoch > 100:
#         lrate = 0.0003
#     return lrate


def show_images():
    """
    Show samples from each class
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_train.shape
    num_classes = len(np.unique(y_train))

    # If using tensorflow, set image dimensions order
    if K.backend() == 'tensorflow':
        K.common.set_image_dim_ordering("th")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(8, 3))

    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        x_idx = X_train[idx,::]
        img_num = np.random.randint(x_idx.shape[0])
        im = np.transpose(x_idx[img_num,::], (1, 2, 0))
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
    if diff > 3600:   # hour
        hours = math.floor(diff / 3600)
        diff = diff - hours * 3600
    if diff > 60:     # minute
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

    s_timestamp = "{0}{1:02d}{02:02d}-{3:02d}{4:02d}{5:02d}" \
                    .format(today.year, today.month, today.day,
                            today.hour, today.minute, today.second)

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
    accuracy = float(num_correct)/result.shape[0]
    return accuracy * 100


def plot_model_history(model_history):
    """
    Plot model accuracy and loss
    @param model_history:
    @return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1), len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    file_name = "model-history-" + get_timestamp() + ".png"

    plt.savefig(file_name)  # Save plot to file
    # plt.show()            # Show plot
    plt.clf()               # Clear current figure
    plt.close(fig)


def load_data():
    # Load data from file
    npzfile = np.load('cifar10.npz')
    # print(npzfile.files)

    X_train = npzfile['X_train']
    X_valid = npzfile['X_valid']
    X_test = npzfile['X_test']

    y_train = npzfile['y_train_hot']
    y_valid = npzfile['y_valid_hot']
    y_test = npzfile['y_test_hot']

    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_test.shape
    num_classes = y_train.shape[1]
    # num_classes = len(np.unique(y_train))
    print("load_data:")
    print("X_train.shape:", num_train, img_channels, img_rows, img_cols)
    print()

    # Reduce datasets to improve performance
    num_records = 2000
    X_train = X_train[:num_records]
    X_valid = X_valid[:num_records]
    X_test = X_test[:num_records]

    y_train = y_train[:num_records]
    y_valid = y_valid[:num_records]
    y_test = y_test[:num_records]

    # num_classes = y_train.shape[1]
    # print(f"num_classes = {num_classes}")
    # print(f"X_test shape: {X_valid.shape}")
    # print(f"y_test shape: {y_valid.shape}")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_data_linux():
    # Create train/test/validation split
    # We want to split our dataset into separate training and test datasets
    # We use the training dataset to fit the model and the test dataset to evaluate
    # its performance to generalize to unseen data.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')

    # Standardize the columns
    # We need to standardize the columns before we feed them to a linear classifier,
    # but if the X values are in the range 0-255 then we can transform them to [0,1].
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0

    # One-hot encoding
    # Represent each integer value as a binary vector that is all zeros
    # except the index of the integer.
    y_train_hot = np_utils.to_categorical(y_train)
    y_test_hot = np_utils.to_categorical(y_test)
    y_valid_hot = np_utils.to_categorical(y_valid)

    print(f"\nnum_classes = {y_train_hot.shape[1]}")

    print(X_train.shape[0], 'Train samples')
    print(X_valid.shape[0], 'Validation samples')
    print(X_test.shape[0], 'Test samples')

    print('\nX_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('X_test shape:', X_test.shape)

    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)
    print('y_test shape:', y_test.shape)

    print('\ny_train_hot shape:', y_train_hot.shape)
    print('y_valid_hot shape:', y_valid_hot.shape)
    print('y_test_hot shape:', y_test_hot.shape)

    # Data Sanity Check
    print(f"\nnum_classes = {y_test_hot.shape[1]}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test:\n{y_test[0:10]}")        # Check that dataset has been randomized

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


# def load_data_linux():
#     # Load data from file
#     npzfile = np.load('cifar10.npz')
#     # print(npzfile.files)
#
#     X_train = npzfile['X_train']
#     X_valid = npzfile['X_valid']
#     X_test = npzfile['X_test']
#
#     y_train = npzfile['y_train_hot']
#     y_valid = npzfile['y_valid_hot']
#     y_test = npzfile['y_test_hot']
#
#     # Reduce datasets to improve performance
#     # num_records = 2000
#     # X_train = X_train[:num_records]
#     # X_valid = X_valid[:num_records]
#     # X_test = X_test[:num_records]
#     #
#     # y_train = y_train[:num_records]
#     # y_valid = y_valid[:num_records]
#     # y_test = y_test[:num_records]
#
#     num_train, img_rows, img_cols, img_channels = X_train.shape
#     num_test, _, _, _ = X_test.shape
#     num_classes = y_train.shape[1]
#     # num_classes = len(np.unique(y_train))
#     print("X_train.shape:", num_train, img_channels, img_rows, img_cols)
#     print()
#
#     # Convert from NCHW to NHWC
#     @tf.function
#     def transform(x):
#         y = tf.transpose(x, [0, 3, 1, 2])
#         return y
#
#     X_train = transform(X_train)
#     X_valid = transform(X_valid)
#     X_test = transform(X_test)
#     print("After transform:")
#     print("X_train.shape:", X_train.get_shape())  # the shape of out is [2000, 32, 32, 3]
#     print("X_valid.shape:", X_valid.get_shape())
#     print("X_test.shape:", X_test.get_shape())
#
#     return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def preprocess():
    """
    Data pre-processing

    @return:
    """
    # Create train/test/validation split
    # We want to split our dataset into separate training and test datasets
    # We use the training dataset to fit the model and the test dataset to evaluate
    # its performance to generalize to unseen data.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')

    # Standardize the columns
    # We need to standardize the columns before we feed them to a linear classifier,
    # but if the X values are in the range 0-255 then we can transform them to [0,1].
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0

    # One-hot encoding
    # Represent each integer value as a binary vector that is all zeros
    # except the index of the integer.
    y_train_hot = np_utils.to_categorical(y_train)
    y_test_hot = np_utils.to_categorical(y_test)
    y_valid_hot = np_utils.to_categorical(y_valid)

    # y_categories = np.unique(y_train)

    print(f"\nnum_classes = {y_train_hot.shape[1]}")

    print(X_train.shape[0], 'Train samples')
    print(X_valid.shape[0], 'Validation samples')
    print(X_test.shape[0], 'Test samples')

    print('\nX_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('X_test shape:', X_test.shape)

    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)
    print('y_test shape:', y_test.shape)

    print('\ny_train_hot shape:', y_train_hot.shape)
    print('y_valid_hot shape:', y_valid_hot.shape)
    print('y_test_hot shape:', y_test_hot.shape)

    # Data Sanity Check
    print(f"\nnum_classes = {y_test_hot.shape[1]}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test:\n{y_test[0:10]}")        # Check that dataset has been randomized

    # Save datasets to file
    np.savez('cifar10_test.npz',
             X_train=X_train, X_valid=X_valid, X_test=X_test,
             y_train=y_train, y_valid=y_valid, y_test=y_test,
             y_train_hot=y_train_hot, y_valid_hot=y_valid_hot, y_test_hot=y_test_hot)


def create_model(name, num_classes):
    """
    Create model for 70% accuracy
    The parameters needed to be different to run on linux.
    @return:
    """
    # epochs = 200
    # batch_size = 128
    weight_decay = 1e-4

    # We expect our inputs to be RGB images of arbitrary size
    inputs = keras.Input(shape=(3, 32, 32))

    # Create the model
    x = Sequential()(inputs)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu", kernel_constraint=max_norm(3))(x)
    x = Dropout(0.2)(x)

    # Finally, we add a classification layer.
    # The activation is softmax which makes the output sum up to 1
    # so the output can be interpreted as probabilities.
    # The model will then make its prediction based on which option has a higher probability.
    outputs = Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    # We can print a summary of how your data gets transformed at each stage of the model.
    # This is useful for debugging.
    model.summary()

    # You can also plot the model as a graph.
    # keras.utils.plot_model(model, "my_first_model.png")

    return model


def fit_model(model, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    @param model:
    @return:
    """
    # lr_init = 0.01
    # decay = lrate / epochs

    # lr_init = 0.1
    # lr_schedule = ExponentialDecay(lr_init, decay_steps=100000,
    #                                decay_rate=0.96, staircase=True)

    # lr_init = 0.1
    # decay_steps = 1.0
    # decay_rate = 0.5
    # lr_schedule = InverseTimeDecay(lr_init, decay_steps, decay_rate, staircase=False, name=None)

    # lr_start = 0.1
    # lr_end = 0.01
    # decay_steps = 10000
    # lr_schedule = PolynomialDecay(lr_start, decay_steps, lr_end, power=0.5)

    # sgd = SGD(learning_rate=lr_schedule)
    # sgd = SGD(learning_rate=lr_schedule, momentum=0.9)

    # Compile the model
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Train the model
    start = time.time()
    # The history object which records what happened over the course of training.
    # The history.history dict contains per-epoch timeseries of metrics values.
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_valid, y_valid), verbose=1)
    end = time.time()
    print(f"\nModel took [{get_elapsed_time(start, end)}] to train")

    # Save the model architecture to disk
    # model_json = model.to_json()
    # with open('model_cnn_80.json', 'w') as json_file:
    #     json_file.write(model_json)

    # Save the model weights
    # model.save_weights('model_cnn_80.h5')

    # Save the whole model (architecture + weights + optimizer state)
    # model.save('model_cnn_80.h5')  # creates a HDF5 file
    # del model  # deletes the existing model

    # Plot model history
    # plot_model_history(history)


def data_augment():
    """
    Add data augmentation for 86% accuracy

    @return:
    """
    epochs = 200
    batch_size = 128

    # Load data from file
    npzfile = np.load('cifar10.npz')
    print(npzfile.files)

    X_train = npzfile['X_train']
    X_valid= npzfile['X_valid']
    y_train = npzfile['y_train_hot']
    y_valid = npzfile['y_valid_hot']

    # Return a compiled model (identical to the previous one)
    model = load_model('model_cnn_70.h5')

    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

    # Train the model
    start = time.time()
    model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                     samples_per_epoch=X_train.shape[0], epochs=epochs,
                                     validation_data=(X_valid, y_valid), verbose=1)
    end = time.time()
    print(f"Model took [{get_elapsed_time(start, end)}] to train (data augmentation)")

    # Save the model architecture to disk
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model_json = model.to_json()
    with open('model_cnn_86.json', 'w') as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save_weights('model_cnn_86.h5')

    # Save the whole model (architecture + weights + optimizer state)
    model.save('model_cnn_86.h5')  # creates a HDF5 file

    # Plot model history
    plot_model_history(model_info)

    # Compute validation accuracy
    print(f"Accuracy on validation data is: {accuracy(X_valid, y_valid, model): .2f} %0.2f")


def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Return a compiled model (identical to the previous one)
    # model = load_model('model_cnn_80.h5')

    # Training
    scores = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    print(f"Train result: accuracy: {scores[1]:.4f}  loss: {scores[0]:.4f}")

    # Testing
    loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(f"Test result: accuracy: {acc:.4f}  loss: {loss:.4f}")

    # x = dict(zip(model.metrics_names, scores))
    # pprint(x, compact=True)

    # Compute test accuracy
    # print(f"\nAccuracy on test data is: {accuracy(X_test, y_test, model):.2f}%")

    # Generate predictions (probabilities -- the output of the last layer) on new data.
    # print("\nGenerate predictions for 10 samples")
    # pred = model.predict(X_test[:10])
    # obs = y_test[:10]

    # print("predictions shape:", predictions.shape)
    # pprint(pred, compact=True)
    # pprint(obs)
    #
    # for i in range(10):
    #     y_pred = np.argmax(pred[i])
    #     y_obs = np.argmax(obs[i])
    #     print(f"(y_pred, y_obs) is {y_pred, y_obs}")


def main():
    train_ds, test_ds, valid_ds = load_data()
    # train_ds, test_ds, valid_ds = load_data_linux()

    X_train, y_train = train_ds
    X_test, y_test = test_ds
    X_valid, y_valid = valid_ds

    num_classes = y_train.shape[1]

    # Get script filename instead of path
    file_name = os.path.basename(sys.argv[0])
    msg = file_name

    # Initialize parser
    parser = argparse.ArgumentParser(description=msg)

    # Add optional arguments
    parser.add_argument("-i", "--images", dest='images',
                        action="store_true", help="Show images")
    parser.add_argument("-p", "--preprocess", dest='preprocess',
                        action="store_true", help="Preprocess data")
    parser.add_argument("-c", "--create", dest='create',
                        action="store_true", help="Create model")
    parser.add_argument("-a", "--augment", dest='augment',
                        action="store_true", help="Data augmentation model")
    parser.add_argument("-e", "--evaluate", dest='evaluate',
                        action="store_true", help="Evaluate model")

    # Read arguments from command line
    args = parser.parse_args()

    # The default image data format convention is different on macOS and linux.
    # print(tf.keras.backend.image_data_format())
    # tf.keras.backend.set_image_data_format('channels_first')

    if args.images:
        show_images()
    elif args.preprocess:
        preprocess()
    elif args.create:
        create_model()
    elif args.evaluate:
        evaluate_model()
    else:
        model = create_model("cifar10_model", num_classes)
        fit_model(model, X_train, X_valid, y_train, y_valid)
        evaluate_model(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    print()
    main()
    print("\nDone!")

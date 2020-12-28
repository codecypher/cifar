#!/usr/bin/env python3
"""
  cifar10_70.py

  Accuracy: 73.40%

  References:
    http://parneetk.github.io/blog/cnn-cifar10/
    https://github.com/abhijeet3922/Object-recognition-CIFAR-10
    https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
"""
import numpy as np
import pandas as pd
import math
import os
import sys
import time
import argparse

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.utils as np_utils

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD


def show_images():
    """
    Show samples from each class
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_train.shape
    num_classes = len(np.unique(y_train))

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
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    file_name = "model-history.png"

    plt.savefig(file_name)  # Save plot to file
    # plt.show()            # Show plot
    plt.clf()               # Clear current figure
    plt.close(fig)


def preprocess():
    """
    Data preprocessing
    @return:
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

    # X_train = X_train[:500]
    # X_valid = X_valid[:500]
    # y_train = y_train[:500]
    # y_valid = y_valid[:500]

    # Create a grid of 3x3 images
    # for i in range(0, 9):
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(Image.fromarray(np.rollaxis(X_train[i],0,3),'RGB'))

    # Show the plot
    # plt.show()

    K.common.set_image_dim_ordering('th')
    seed = 7
    np.random.seed(seed)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')

    # Standardize the columns
    # Sometimes we need to standardize the columns before we feed them to a linear classifier,
    # but if the X values are in the range 0-255 then we can transform them to [0,1].
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0

    # Standardize the columns (Z-score)
    # We need to standardize the columns before we feed them to a linear classifier
    # mean = np.mean(X_train, axis=(0, 1, 2, 3))
    # std = np.std(X_train, axis=(0, 1, 2, 3))
    # X_train = (X_train - mean) / (std + 1e-7)
    # X_test = (X_test - mean) / (std + 1e-7)
    # X_valid = (X_valid - mean) / (std + 1e-7)

    # Same thing
    # stdscaler = preprocessing.StandardScaler().fit(X_train)
    # X_train_scaled = stdscaler.transform(X_train)
    # X_test_scaled  = stdscaler.transform(X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_valid = np_utils.to_categorical(y_valid)

    print(f"num_classes = {y_test.shape[1]}")

    print(X_train.shape[0], 'Train samples')
    print(X_valid.shape[0], 'Validation samples')
    print(X_test.shape[0], 'Test samples')

    print('\nX_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('X_test shape:', X_test.shape)

    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)
    print('y_test shape:', y_test.shape)

    # Data Sanity Check
    print(f"\nnum_classes = {y_test.shape[1]}")
    print(f"X_test shape: {X_test.shape}")
    print(y_test[0:10])        # Check that dataset has been randomized

    # print('\nX_train:\n', X_train[:1,:,:,:])

    # Save datasets to file
    np.savez('cifar10_70.npz',
             X_train=X_train, X_valid=X_valid, X_test=X_test,
             y_train=y_train, y_valid=y_valid, y_test=y_test)


def create_model():
    epochs = 25
    batch_size = 32
    lrate = 0.01
    decay = lrate / epochs

    # Load datasets from file
    npzfile = np.load('cifar10_70.npz')
    print(npzfile.files)

    X_train = npzfile['X_train']
    X_valid = npzfile['X_valid']
    X_test = npzfile['X_test']

    y_train = npzfile['y_train']
    y_valid = npzfile['y_valid']
    y_test = npzfile['y_test']

    num_classes = y_test.shape[1]
    print(f"num_classes = {num_classes}")
    print(f"X_test shape: {X_test.shape}")

    # Create model
    # Conv2D                - 2D convolution layer (e.g. spatial convolution over images).
    # Activation            - Applies an activation function to an output.
    # ELU                   - Exponential Linear Unit (Advanced Activation Layers)
    # BatchNormalization    - Batch normalization layer
    # MaxPooling2D          - Max pooling operation for spatial data.
    # Dropout               - Applies Dropout to the input.
    # Flatten               - Flattens the input. Does not affect the batch size.

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(3, 32, 32), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    start = time.time()
    # Train the model
    model_info = model.fit(X_train, y_train,
                           validation_data=(X_valid, y_valid),
                           epochs=epochs, batch_size=batch_size)
    end = time.time()
    print(f"Model took [{get_elapsed_time(start, end)}] to train")
    # print(f"Model took {(end - start):.2f} seconds to train")
    # print("Model took %0.2f seconds to train" % (end - start))

    # Save the model architecture to disk
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model_json = model.to_json()
    with open('model_cnn_70.json', 'w') as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save_weights('model_cnn_70_weights.h5')

    # Save the whole model (architecture + weights + optimizer state)
    model.save('model_cnn_70.h5')  # creates a HDF5 file
    # del model  # deletes the existing model

    # Plot model history
    plot_model_history(model_info)


def evaluate_model():
    batch_size = 128

    # Load data from file
    npzfile = np.load('cifar10_70.npz')
    print(npzfile.files)

    X_train = npzfile['X_train']
    X_test = npzfile['X_test']
    y_train = npzfile['y_train']
    y_test = npzfile['y_test']

    # load json and create model
    # json_file = open('models/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)

    # load weights into new model
    # model.load_weights("models/model.h5")
    # print("Loaded model from disk")

    # Return a compiled model (identical to the previous one)
    model = load_model('model_cnn_70.h5')

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
    parser.add_argument("-i", "--images", dest='images',
                        action="store_true", help="Show images")
    parser.add_argument("-p", "--preprocess", dest='preprocess',
                        action="store_true", help="Preprocess data")
    parser.add_argument("-c", "--create", dest='create',
                        action="store_true", help="Create model")
    parser.add_argument("-e", "--evaluate", dest='evaluate',
                        action="store_true", help="Evaluate model")

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

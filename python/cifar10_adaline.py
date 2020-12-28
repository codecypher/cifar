# cifar10_adaline.py

# ADALINE Model

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, \
    accuracy_score, precision_score, recall_score
from sklearn.linear_model import Perceptron

# Load datasets from file
npzfile = np.load('cifar10.npz')
print(npzfile.files)

x_train = npzfile['x_train']
x_test = npzfile['x_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']

# Standardize the columns
x_train = x_train / 255
x_test = x_test / 255

# The model cannot deal with 2D array so we have to convert to 1D array.
x_train_flat = np.empty(shape=[x_train.shape[0]] + [3072], dtype='float32')

for i in range(x_train.shape[0]):
    x_train_flat[i,:] = x_train[i,:,:].flatten()

# Flatten x_test array
x_test_flat = np.empty(shape=[x_test.shape[0]] + [3072], dtype='float32')
for i in range(x_test.shape[0]):
    x_test_flat[i,:] = x_test[i,:,:].flatten()

x_train = x_train_flat
x_test = x_test_flat
y_train = y_train.ravel()
y_test = y_test.ravel()

print('\n', type(x_train))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


ppn = Perceptron(eta0=0.1,max_iter=10)
start = time.time()
ppn.fit(x_train, y_train)
end = time.time()

predicted = ppn.predict(x_test)
predicted[0:5]

print("\nModel took %0.2f seconds to train"%(end - start))

print('\nAccuracy', accuracy_score(y_test, predicted))
print('\nADALINE Model Results\n\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
print('\n Classifcation Report')
print(classification_report(y_test, predicted))
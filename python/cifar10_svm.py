# cifar10_svm.py

# Support Vector Machine (SVM)

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import model_selection
from scipy.io import loadmat

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import hinge_loss
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def run_svc(svc, title):
    # Fit model
    start = time.time()
    svc.fit(x_train, y_train)
    end = time.time()
    print("\nModel took %0.2f seconds to train"%(end - start))

    # Calculate predictions
    start = time.time()
    predicted = svc.predict(x_test)
    end = time.time()
    print("Model took %0.2f seconds to calculate predictions"%(end - start))

    # Output results
    print('\naccuracy', accuracy_score(y_test, predicted))
    print('\nSVM Results for ' + title)
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, predicted))

    print('\nClassification Report:', classification_report(y_test, predicted))
    #print("Hinge loss", hinge_loss(y_test, predicted))


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


# Linear
svc = SVC(kernel='linear', C=1)
run_svc(svc, 'Linear')

# Radial Basis Function (RBF)
svc = SVC(kernel='rbf', gamma=1, C=1)
run_svc(svc, 'Radial Basis Function (RBF)')

# Polynomial
svc = SVC(kernel='poly', degree=5, C=1)
run_svc(svc, 'Polynomial)')


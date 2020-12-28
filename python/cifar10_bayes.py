# cifar10_bayes.py

# Multinomial Naive Bayes

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from mlclass2 import simplemetrics, plot_decision_2d_lda
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# Load datasets from file
npzfile = np.load('cifar10.npz')
print(npzfile.files)

x_train = npzfile['x_train']
x_test = npzfile['x_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']

# Standardize the columns (z-score)
# We need to standardize the columns before we feed them to a linear classifier
x_train = x_train / 255.0
x_test = x_test / 255.0

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

# Run MNB model
start = time.time()
train_class = MultinomialNB().fit(x_train, y_train)
end = time.time()

predicted = train_class.predict(x_test)

print('predicted:', predicted[0:10])

print('\nx_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('predicted shape:', predicted.shape)

#simplemetrics(y_test,p redicted)
#plot_decision_2d(gnb, x_train,y,title="Full Data Set")

print("\nModel took %0.2f seconds to train"%(end - start))

print('\nAccuracy', accuracy_score(y_test, predicted))
print('\nMultinomial Naive Bayes Results\n\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))
print('\n Classifcation Report')
print(classification_report(y_test, predicted))

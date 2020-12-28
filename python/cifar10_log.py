# cifar10_log.py

# Multinomial Stochastic Logistic Regression

import matplotlib.pyplot as plt
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
from keras.optimizers import SGD
from sklearn.metrics import classification_report, f1_score, confusion_matrix, \
    accuracy_score, precision_score, recall_score

# Load datasets from file
npzfile = np.load('cifar10.npz')
print(npzfile.files)

x_train = npzfile['x_train']
x_test = npzfile['x_test']
y_train = npzfile['y_train_hot']
y_test = npzfile['y_test_hot']


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

# One-hot encoding
# Convert categorical variable into dummy/indicator variables
# There are 10 classes of characters we are trying to identify (A-J)
#y_train = y_train_hot
#y_test = y_test_hot


# Stochastic Logistic Regression
model = Sequential()

# validation loss
model.add(Dense(output_dim=10, input_shape=[3072], activation='softmax', W_regularizer=l2(0.05)))

# Compile model (3072*10 = 30720 parameters)
sgd = SGD(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.summary()

start = time.time()

# Fit the model
history = model.fit(x_train, y_train, batch_size = 128, epochs = 1000, verbose=1, validation_data=(x_test, y_test))


end = time.time()


# Save history to a file
#np.savez('cifar10_log.npz', history=history)


# Saving the model architecture
model_json = model.to_json()
with open('model_log.json', 'w') as json_file:
    json_file.write(model_json)

# Saving the model weights
model.save_weights('model_log.h5')


print("Model took %0.2f seconds to train"%(end - start))


# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'g--')
plt.title('Logistic Regression Model Loss')
plt.ylabel('Cross-Entropy')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')

print("Loss after final iteration: ", history.history['val_loss'][-1])

plt.show()

# cifar

The goal of this assignment is to use the keras deep learning package in Python for implementing various artificial neural networks (ANNs) with the CIFAR10 dataset.

The dataset used here is the CIFAR10 dataset included with Keras [1][2]. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

I prefered using the built-in `cifar10.load_data()` function rather than manually downloading and processing the files. Therefore, I chose to use the default test dataset of 10K images and split the 50K training images evenly between train and validation datasets. First, I ran the ANN models using the optimal parameter settings with the 25K training and 25K validation datasets. 

## Data Preprocessing

Neural networks need to process vectorized and standardized datasets rather than raw text files (CSV) or image files (JPEG). Fortunately, the built-in function `cifar10.load_data()` did most of the "heavy-lifting" of data cleaning and wrangling to manipulate the dataset into a clean form that could be used by Keras. Fortunately, the dataset was fairly "clean". I did not find any duplicate or irrelevant entries, typos, or mislabeled classes, etc. 

It is best practice to standardize the columns via feature normalization when using a linear classifier for binary classification. Since the images (X_train, X_test, X_valid) consisted of RGB values in the range 0-255, we can transform them to [0, 1] by dividing by 255. 

The data also appeared to be randomly-selected so I was able to skip that step.

The class values (y_train, y_test, y_valid) are in the range 0-9, so I applied one-hot encoding to represent each integer value as a binary vector that is all zeros except the index of the orginal class (integer) value (say [0 0 0 0 0 0 0 1 0 0] instead of 8). It turns out the Keras provides some built-in utilities (tensorflow.keras.utils) for handling NumPy arrays as well as other formats. The Keras documentation mentions that is best practice to make data pre-processing part of the machine learning model [3]. However, I am not concerned with portabiliity, so I chose to save the "cleaned" data to disk (`*.npz`) using the NumPy `savez()` function rather than run the pre-processing every time the program executed.

## References

[1] Tensorflow 2.0 (includes Keras), https://keras.io/, 2.0.0. 2020.

[2] A. Krizhevsky, 2009, "The CIFAR-10 dataset," Accessed: July 5, 2020. [Online]. Available: https://www.cs.toronto.edu/~kriz/cifar.html

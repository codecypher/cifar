# cifar

This repo containes the source code for my AI graduate course project using the Keras deep learning package in Python for implementing various artificial neural networks (ANNs) with the CIFAR-10 dataset.

The dataset used here is the CIFAR10 dataset included with Keras [1][2]. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

I prefered using the built-in `cifar10.load_data()` function rather than manually downloading and processing the files. Therefore, I chose to use the default test dataset of 10K images and split the 50K training images evenly between train and validation datasets. First, I ran the ANN models using the optimal parameter settings with the 25K training and 25K validation datasets. 

## Data Preprocessing

Neural networks need to process vectorized and standardized datasets rather than raw text files (CSV) or image files (JPEG). Fortunately, the built-in function `cifar10.load_data()` did most of the "heavy-lifting" of data cleaning and wrangling to manipulate the dataset into a clean form that could be used by Keras. Fortunately, the dataset was fairly "clean". I did not find any duplicate or irrelevant entries, typos, or mislabeled classes, etc. 

It is best practice to standardize the columns via feature normalization when using a linear classifier for binary classification. Since the images (X_train, X_test, X_valid) consisted of RGB values in the range 0-255, we can transform them to [0, 1] by dividing by 255. 

The data seemed to be randomly-selected so I was able to skip that step.

The class values (y_train, y_test, y_valid) are in the range 0-9, so I applied one-hot encoding to represent each integer value as a binary vector that is all zeros except the index of the orginal class (integer) value (say [0 0 0 0 0 0 0 1 0 0] instead of 8). It turns out the Keras provides some built-in utilities (tensorflow.keras.utils) for handling NumPy arrays as well as other formats. 

The Keras documentation mentions that is best practice to make data pre-processing part of the machine learning model [3]. However, I am not concerned with portabiliity, so I chose to save the "cleaned" data to disk (`*.npz`) using the NumPy `savez()` function rather than run the pre-processing every time the program executed.

Since the program needed to be run numerous times for this experiment, I chose to use the Python `argparse` module [4] to process separate command-line arguments as needed: display a sample of the images (-i), preprocess the data and save to disk (-p), run code to create the ANN model (-c), and run code to evaluate the ANN model (-e). If no command-line argument was given, the program would run the default process: create the model, fit the model, and evaluate the model. This made the experimentation process much easier and fluid as I tweaked different parameters of the model(s).

For most coding projects, I also create utility functions to make it easier to complete a task. First, I created the `plot_model_history` function to plot the model accuracy and loss using matplotlib. Next, I created a nice function to compute elapsed time called `get_elapsed_time` to make it easier to display the elapsed time when needed (primarily the amount of time needed to "fit" or train the model). Finally, I also created the `get_timestamp` function to generate a timestamp (YYYYMMDD-HHMMSS) to append to the graphs generated when calling `plot_model_history()`.

When using Keras to create ANN models, there are an infinite number of possible choices for layers and/or parameters. I experimented with many possible models, but I also did some research and found three good models for experimentation, each achieving 70%, 80%, and 90% accuracy [5][6][7]. After experimenting with the three models, I decided to choose the second model [6] since the Keras ANN model was simpler and easier use when experimenting with the different model parameters.

One technique that is often used for image classification is image rescaling and cropping, or data augmentation. The ANN model that achieved 90% accuracy implemented data augmentation. However, due to time constraints and to keep the model simple for experimentation (KISS), I chose to omit data augmentation from the model.

## The Model

The core data structures of Keras are _layers_ and _models_. The simplest type of model is the `Sequential` model which is a linear stack of layers. For more complex architectures, it is recommended to use the Keras functional API which allows you to build arbitrary graphs of layers, or write models entirely from scratch via subclasssing [8]. The model I am using is pretty simple. However, I found the syntax of the Functional API to be easier to use and more intuitive. Therefore, I chose to implement the ANN model using the Keras Functional API.

The purpose of the loss function is to compute the quantity that a model should try to minimize during training. The Keras API Documentation recommends using the crossentropy loss function when there are two or more label classes, and it is the most common choice for classification. This is done by setting the loss argument on the compile() method to "categorical_crossentropy". A lower score for the crossentropy loss function indicates that the model is performing better.

## The Experiments

The primary goal of the assignment was to build a neural network model with various parameters and evaluate its performance on the validation data. To make the problem tractable, I chose to reduce the number of images in the datasets to 2000 images each (training, test, and validation), the number of epochs (iterations) to 20, and the batch size to 64. Since there were so many parameters to evaluate, there was not time to properly run the program dozens or even hundreds of times and compute the average accuracy and loss values (similar to previous assignments). However, I was able to run a majority of the experiments several times to rule out the possibility of anomalies in the model and/or bugs in the program.

I ran the program several times using the model shown in Figure 1 on the reduced dataset of 6K images to obtain a baseline for the experiment. An example of the program results is shown in Figure 2 and a summary of the ANN model is shown in Figure 3. I also ran the program one time using the full dataset of 50K images (training=20K, validation=20K, and test=10K), epochs = 200, and batch_size = 129. The result was a test accuracy of approximately 74%. The plots of the model accuracy and loss per epoch are shown in Figure 4.

Then, I experimented with the following parameters: number of hidden layers, number of units per layer, activation functions, regularization techniques, and learning rate.


## References

[1] Tensorflow 2.0 (includes Keras), https://keras.io/, 2.0.0. 2020.

[2] A. Krizhevsky, 2009, "The CIFAR-10 dataset," Accessed: July 5, 2020. [Online]. Available: https://www.cs.toronto.edu/~kriz/cifar.html

[3] F. Chollet, "Introduction to Keras for Engineers," Keras Documentation. Accessed: July 5, 2020. [Online]. Available: https://keras.io/getting_started/intro_to_keras_for_engineers/

[4] N. Aggarwal, "Command Line Arguments in Python," GeeksforGeeks. Accessed: July 5, 2020. [Online]. Available: https://www.geeksforgeeks.org/command-line-arguments-in-python/


[5] A. Kumar, "Object-recognition-CIFAR-10," GitHub. Accessed: July 5, 2020. [Online]. Available: https://github.com/abhijeet3922/Object-recognition-CIFAR-10/blob/master/cifar10.py

[6] P. Kaur, "Convolutional Neural Networks (CNN) for CIFAR-10 Dataset," GitHub. Accessed: July 5, 2020. [Online]. Available: http://parneetk.github.io/blog/cnn-cifar10/

[7] A. Kumar, "Achieving 90% accuracy in Object Recognition Task on CIFAR-10 Dataset with Keras: Convolutional Neural Networks," Machine Learning in Action. Accessed: July 5, 2020. [Online]. Available: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/


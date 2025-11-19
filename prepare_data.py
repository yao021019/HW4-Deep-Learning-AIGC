import numpy as np
import tensorflow as tf

def load_and_preprocess_data():
    """
    Loads the MNIST dataset, preprocesses it, and saves it to a file.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Save the preprocessed data
    np.savez('mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    print("MNIST data has been preprocessed and saved to mnist_data.npz")

if __name__ == '__main__':
    load_and_preprocess_data()

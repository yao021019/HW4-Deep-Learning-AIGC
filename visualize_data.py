import numpy as np
import matplotlib.pyplot as plt

def visualize_mnist_digit():
    """
    Loads the preprocessed MNIST data and displays a single digit.
    """
    # Load the preprocessed data
    data = np.load('mnist_data.npz')
    x_train = data['x_train']

    # Reshape the first image to 28x28
    first_image = x_train[0].reshape(28, 28)

    # Display the image
    plt.imshow(first_image, cmap='gray')
    plt.title("A digit from the MNIST dataset")
    plt.show()

if __name__ == '__main__':
    visualize_mnist_digit()

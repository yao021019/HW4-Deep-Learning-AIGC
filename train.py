import numpy as np
import tensorflow as tf
from model import NeuralNetwork
from sklearn.metrics import accuracy_score

def train_model():
    """
    Loads the MNIST dataset, preprocesses it, trains the neural network,
    evaluates its accuracy, and saves the trained model.
    """
    # Load the MNIST dataset
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train_raw.astype('float32') / 255.0
    x_test = x_test_raw.astype('float32') / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=10)

    # Define model parameters
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    n1 = 128  # Number of neurons in the first hidden layer
    n2 = 64   # Number of neurons in the second hidden layer
    n3 = 32   # Number of neurons in the third hidden layer
    epochs = 50
    learning_rate = 0.1
    batch_size = 64

    # Initialize the neural network
    nn = NeuralNetwork(input_size, n1, n2, n3, output_size)

    # Train the network
    print("Training the neural network...")
    nn.train(x_train, y_train, epochs, learning_rate, batch_size)
    print("Training finished.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred_encoded = nn.predict(x_test)
    y_test_labels = y_test_raw # Use raw labels for accuracy_score
    accuracy = accuracy_score(y_test_labels, y_pred_encoded)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    nn.save_model('model.npz')

if __name__ == '__main__':
    train_model()

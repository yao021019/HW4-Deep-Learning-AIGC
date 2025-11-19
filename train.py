import numpy as np
from model import NeuralNetwork
from sklearn.metrics import accuracy_score

def train_model():
    """
    Loads the preprocessed MNIST data, trains the neural network,
    evaluates its accuracy, and saves the trained model.
    """
    # Load the preprocessed data
    data = np.load('mnist_data.npz')
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

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
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_encoded)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    nn.save_model('model.npz')

if __name__ == '__main__':
    train_model()

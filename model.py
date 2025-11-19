import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, n1, n2, n3, output_size):
        """
        Initializes the neural network with three hidden layers.
        """
        self.W1 = np.random.randn(input_size, n1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, n1))
        self.W2 = np.random.randn(n1, n2) * np.sqrt(2. / n1)
        self.b2 = np.zeros((1, n2))
        self.W3 = np.random.randn(n2, n3) * np.sqrt(2. / n2)
        self.b3 = np.zeros((1, n3))
        self.W4 = np.random.randn(n3, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        """
        Softmax activation function.
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        """
        Performs the forward pass through the network.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = self.softmax(self.z4)
        return self.a4

    def backward(self, X, y, output):
        """
        Performs the backward pass and calculates gradients.
        """
        self.d_W = {}
        self.d_b = {}
        
        error = output - y
        d_output = error / len(X)

        error_N3 = d_output.dot(self.W4.T)
        d_N3 = error_N3 * self.relu_derivative(self.z3)

        error_N2 = d_N3.dot(self.W3.T)
        d_N2 = error_N2 * self.relu_derivative(self.z2)

        error_N1 = d_N2.dot(self.W2.T)
        d_N1 = error_N1 * self.relu_derivative(self.z1)

        self.d_W[4] = self.a3.T.dot(d_output)
        self.d_b[4] = np.sum(d_output, axis=0, keepdims=True)
        self.d_W[3] = self.a2.T.dot(d_N3)
        self.d_b[3] = np.sum(d_N3, axis=0, keepdims=True)
        self.d_W[2] = self.a1.T.dot(d_N2)
        self.d_b[2] = np.sum(d_N2, axis=0, keepdims=True)
        self.d_W[1] = X.T.dot(d_N1)
        self.d_b[1] = np.sum(d_N1, axis=0, keepdims=True)

    def update_weights(self, learning_rate):
        """
        Updates the weights and biases of the network.
        """
        self.W1 -= learning_rate * self.d_W[1]
        self.b1 -= learning_rate * self.d_b[1]
        self.W2 -= learning_rate * self.d_W[2]
        self.b2 -= learning_rate * self.d_b[2]
        self.W3 -= learning_rate * self.d_W[3]
        self.b3 -= learning_rate * self.d_b[3]
        self.W4 -= learning_rate * self.d_W[4]
        self.b4 -= learning_rate * self.d_b[4]
        
    def train(self, X, y, epochs, learning_rate, batch_size=64):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Perform forward pass, backward pass, and update weights
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                self.update_weights(learning_rate)

            # Print the loss at the end of each epoch
            if epoch % 10 == 0:
                output = self.forward(X)
                loss = -np.mean(y * np.log(output + 1e-9))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict_proba(self, X):
        """
        Makes predictions on new data and returns the probabilities for each class.
        """
        return self.forward(X)

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_model(self, path):
        """
        Saves the model's weights and biases.
        """
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model's weights and biases.
        """
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        self.W4 = data['W4']
        self.b4 = data['b4']
        print(f"Model loaded from {path}")

if __name__ == '__main__':
    # This part is for testing the model implementation
    
    # Create a dummy dataset
    X_train = np.random.rand(100, 784)
    y_train = np.random.randint(0, 10, 100)
    y_train_one_hot = np.zeros((100, 10))
    y_train_one_hot[np.arange(100), y_train] = 1

    # Initialize and train the network
    nn = NeuralNetwork(input_size=784, n1=20, n2=20, n3=20, output_size=10)
    nn.train(X_train, y_train_one_hot, epochs=1000, learning_rate=0.01)

    # Make a prediction
    X_test = np.random.rand(1, 784)
    prediction = nn.predict(X_test)
    print(f"Prediction: {prediction}")

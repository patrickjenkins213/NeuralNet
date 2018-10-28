import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

import data_util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='mnist_colmajor.mat',
                        help='Path to MNIST pkl.gz file')
    parser.add_argument('--percent_train', type=float, default=.2,
                       help='Percent of training data to use as training vs dev')
    parser.add_argument('--epochs', type=int, default=100,
                   help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1,
                   help='Size of each training batch')
    parser.add_argument('--learning_rate', type=float, default=.5,
                        help='Learning rate for updating the weights')
    return parser.parse_known_args()

class Data:
    def __init__(self, f, percent_train=.8):
        mat = spio.loadmat(f)

        self.full_train_data   = mat['images_train']
        self.full_train_labels = mat['labels_train']

        # divide up the training data into training and dev
        train_indeces = np.random.choice(np.arange(len(self.full_train_data)), int(percent_train * len(self.full_train_data)), replace=False)
        dev_indeces   = list(set(range(len(self.full_train_data))) - set(train_indeces))

        self.train_data   = self.full_train_data[train_indeces]
        self.train_labels = self.full_train_labels[train_indeces].transpose()[0]
        self.dev_data     = self.full_train_data[dev_indeces]
        self.dev_labels   = self.full_train_labels[dev_indeces]
        self.test_data    = mat['images_test']
        self.test_labels  = mat['labels_test']

class NNet:
    def __init__(self):
        self.layers = []

    # TODO: for batching, create train_batch function and call in loop
    def train(self, X, y, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            print(f"Training epoch {epoch}")
            probs = self.forward(X)
            self.backward(y)
            for layer in self.layers:
                # TODO: make more generic with param and d_param collections (dictionaries)
                if isinstance(layer, FullyConnectedLayer):
                    # TODO: create optimizer class
                    layer.weights -= learning_rate * layer.d_weights
                    layer.biases -= learning_rate * layer.d_biases

            loss = self.loss(y)
            print(f"Loss: {loss}")

    def loss(self, y):
        return self.layers[-1].loss(y)

    def evaluate(self, X, y):
        predictions = self.predict_proba(X)
        best = np.argmax(predictions, axis=0)
        matches, = np.where(y.flatten() == best)
        return len(matches) / len(y)

    def predict_proba(self, X):
        return self.forward(X)

    def forward(self, X):
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, y):
        result = self.layers[-1].backward(y)
        for layer in reversed(self.layers[:-1]):
            result = layer.backward(result)
        return result

class FullyConnectedLayer:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = None
        self.biases = None
        self.initialize()

        self.d_weights = None
        self.d_biases = None
        self.inputs = None

    # in future: add alternate initializations, pass as function to __init__
    def initialize(self):
        # Glorot & Bengio 2010
        std = 1 / np.sqrt((self.num_inputs + self.num_outputs) / 2)
        self.weights = np.random.randn(self.num_outputs, self.num_inputs) * std
        self.biases = np.zeros((self.num_outputs, 1))

    def forward(self, X):
        self.inputs = X
        return self.weights.dot(X) + self.biases

    def backward(self, d_outputs):
        self.d_weights = d_outputs.dot(self.inputs.T)
        self.d_biases = np.sum(d_outputs, axis=1, keepdims=True)
        return self.weights.T.dot(d_outputs)

class Softmax:
    def __init__(self):
        self.probs = None

    def forward(self, X):
        maxs = X.max(axis=0, keepdims=True)
        X -= maxs
        exps = np.exp(X)
        self.probs = exps / np.sum(exps, axis=0, keepdims=True)
        return self.probs

    def backward(self, y):
        d_probs = self.probs.copy()
        d_probs[y, np.arange(len(y))] -= 1
        return d_probs / len(y)

    def loss(self, y):
        return -np.mean(np.log(self.probs[y, np.arange(len(y))]))

class Sigmoid:
    def forward(self, X):
        self.outputs = 1 / (1 + np.exp(-X))
        return self.outputs

    def backward(self, d_outputs):
        return d_outputs * (self.outputs * (1 - self.outputs))

class Tanh:
    def forward(self, X):
        self.outputs = np.tanh(X)
        return self.outputs

    def backward(self, d_outputs):
        return d_outputs * (1 - self.outputs ** 2)

class Relu:
    def forward(self, X):
        self.outputs = np.maximum(0, X)
        return self.outputs

    def backward(self, d_outputs):
        return d_outputs * np.where(self.outputs > 0, 1, 0)

def main():
    ARGS, unparsed = parse_args()

    print("Loading data...")
    data = Data(ARGS.datafile, ARGS.percent_train)

    net = NNet()

    net.layers.append(FullyConnectedLayer(784, 200))
    net.layers.append(Relu())
    net.layers.append(FullyConnectedLayer(200, 100))
    net.layers.append(Relu())
    net.layers.append(FullyConnectedLayer(100, 10))
    net.layers.append(Softmax())

    print("Training net...")
    net.train(data.train_data.T, data.train_labels, ARGS.epochs, ARGS.learning_rate, len(data.train_data))

    print("Evaluating net...")
    accuracy = net.evaluate(data.dev_data.T, data.dev_labels)
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    main()

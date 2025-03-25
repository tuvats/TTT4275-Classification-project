import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.5, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuacies = [], []

    def sigmoid_function(self, X):
        return X
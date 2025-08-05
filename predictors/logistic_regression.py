import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinearRegression:
    def __init__(self, rate=0.001, iteration = 1000):
        self.rate = rate
        self.iteration = iteration
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_X = X.shape[1]
        n_y = y.shape[0]

        self.weight = np.zeros(n_X)
        self.bias = 0

        for i in range(self.iteration):
            y_pred = sigmoid(self.bias + np.dot(X, self.weight))

            error = y_pred - y
            dw = (2 / n_y) * np.dot(X.T, error)
            db = (2 / n_y) * np.sum(error)

            self.weight -= self.rate * dw
            self.bias -= self.rate * db

    def predict(self, X):
        return sigmoid(self.bias + np.dot(X, self.weight))

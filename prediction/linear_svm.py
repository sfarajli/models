import numpy as np

class BaseLinearSVM:
    def __init__(self, rate=0.001, iterations=1000, lmbd=0.01):
        self.rate = rate
        self.iterations = iterations
        self.lmbd = lmbd
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        y = self._yconvert(y)

        for _ in range(self.iterations):
            for index, x_i in enumerate(X):
                dw, db = self._get_gradients(x_i, y[index], self.weight, self.bias)
                self.weight -= dw
                self.bias -= db

    def predict(self, X):
        return self._predict(X)

class LinearSVC(BaseLinearSVM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _predict(self, X):
        return np.sign(np.dot(X, self.weight) - self.bias)

    def _yconvert(self, y):
        return np.sign(y)

    def _get_gradients(self, x, y, weight, bias):
        if y * (np.dot(x, weight ) - bias) >= 1:
            dw = 2 * self.lmbd * weight
            db = 0
        else:
            dw = 2 * self.lmbd * weight - y * x
            db = y

        return self.rate * dw, self.rate * db

class LinearSVR(BaseLinearSVM):
    def __init__(self, epsilon=0.1, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def _predict(self, X):
        return np.dot(X, self.weight) - self.bias

    def _yconvert(self, y):
        return y

    def _get_gradients(self, x, y, weight, bias):
        y_pred = np.dot(x, weight) - bias
        error = y_pred - y

        if abs(error) <= self.epsilon:
            dw = 2 * self.lmbd * weight
            db = 0
        else:
            sign = 1 if error > 0 else -1
            dw = 2 * self.lmbd * weight + sign * x
            db = -sign

        return self.rate * dw, self.rate * db

class SVC:
    def __init__(self, rate=0.001, iterations=1000, lmbd=0.01):
        self.rate = rate
        self.iterations = iterations
        self.lmbd = lmbd
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        y_sign = np.sign(y)

        for _ in range(self.iterations):
            for index, x_i in enumerate(X):
                if y_sign[index] * (np.dot(x_i, self.w ) - self.b) >= 1:
                    dw = self.rate * (2 * self.lmbd * self.w)
                    db = 0
                else:
                    dw = self.rate * (2 * self.lmbd * self.w - y_sign[index] * x_i)
                    db = self.rate * y_sign[index]

                self.w -= dw
                self.b -= db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

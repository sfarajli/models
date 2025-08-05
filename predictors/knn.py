import numpy as np
import math
from collections import Counter

def distance(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("Vectors must be the same length")

    return np.linalg.norm(x1 - x2)

class KNN:
    def __init__(self, k=3, mode="classification"):
        self.mode = mode
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        distances = []
        for i, x_train in enumerate(self.X):
            dist = distance(x, x_train)
            distances.append((dist, self.y[i]))

        # sort according to the distances
        sorted_distances = sorted(distances, key=lambda x: x[0])
        k_elements = sorted_distances[:self.k]
        k_targets = [x[1] for x in k_elements]

        if self.mode == "classification":
            count = Counter(k_targets)
            most_common_label, count_occurrences = count.most_common(1)[0]
            return most_common_label
        elif self.mode == "regression":
            return np.mean(k_targets)
        else:
            raise ValueError("Mode must be 'classification' or 'regression'")

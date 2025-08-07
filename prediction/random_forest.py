import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(
        self,
        n_estimators          = 100,
        criterion             = 'gini',
        max_depth             = None,
        min_samples_split     = 2,
        min_samples_leaf      = 1,
        max_features          = None,
        random_state          = None,
        max_leaf_nodes        = None,
        min_impurity_decrease = 0.0,
        bootstrap             = True,
    ):
        self.n_estimators          = n_estimators
        self.criterion             = criterion
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.min_samples_leaf      = min_samples_leaf
        self.max_features          = max_features
        self.random_state          = random_state
        self.max_leaf_nodes        = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap             = bootstrap
        self.forest                = []

        self._rng = np.random.RandomState(random_state) if random_state is not None else np.random

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._get_samples(X, y)

            seed = self._rng.randint(0, 2**32 - 1)
            tree = DecisionTreeClassifier(
                criterion             = self.criterion,
                max_depth             = self.max_depth,
                min_samples_split     = self.min_samples_split,
                min_samples_leaf      = self.min_samples_leaf,
                max_features          = self.max_features,
                random_state          = seed,
                max_leaf_nodes        = self.max_leaf_nodes,
                min_impurity_decrease = self.min_impurity_decrease,
                )
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)

    def _get_samples(self, X, y):
        if self.bootstrap:
            return self._bootstrap_sample(X, y)

        return X, y

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Generate random indices with replacement
        indices = self._rng.choice(n_samples, size=n_samples, replace=True)

        X_sample = X[indices]
        y_sample = y[indices]
        return X_sample, y_sample

    def predict(self, X):
        predictions = [ self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        values = []
        for tree in self.forest:
            values.append(tree.predict(x))

        return Counter(values).most_common(1)[0][0]

import numpy as np
from collections import Counter

# TODO: check for min_samples_split
# TODO: handle all the parameters
# TODO: add regressor

class Node:
    def __init__(self, right=None, left=None, threshold=None, feature=None, label=None):
        self.right = right
        self.left = left
        self.threshold = threshold
        self.feature = feature
        self.label = label

    def isleaf(self):
        if self.label == None:
            return False

        return True

class DecisionTreeClassifier:
    def __init__(
        self,
        criterion             = "gini", # entropy
        splitter              = "best", # random
        max_depth             = None,
        min_samples_split     = 2,               # FIXME: not implemented  # split
        min_samples_leaf      = 1,               # FIXME: not implemented  # split
        max_features          = None,
        random_state          = None,
        max_leaf_nodes        = None,            # FIXME: not implemented  # split
        min_impurity_decrease = 0.0,             # FIXME: not implemented  # split
        class_weight          = None             # FIXME: not implemented  # misc probably
    ):
        self.criterion             = criterion
        self.splitter              = splitter
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.min_samples_leaf      = min_samples_leaf
        self.max_features          = max_features
        self.random_state          = random_state
        self.max_leaf_nodes        = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight          = class_weight
        self._root_node            = None

    def fit(self, X, y):
        self._root_node = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        node = Node()

        if self._impurity(y) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            node.label = self._getlabel(y)
            return node

        feature_inx, threshold = self._best_split_params(X, y)
        if feature_inx == None or threshold == None:
            node.label = self._getlabel(y)
            return node


        X_right, X_left, y_right, y_left = self._split(X, y, feature_inx, threshold)

        node.threshold = threshold
        node.feature  = feature_inx
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def _info_gain(self, X, y, feature_inx, threshold):
        _, _, y_right, y_left = self._split(X, y, feature_inx, threshold)

        n = len(y)
        n_right = len(y_right)
        n_left = len(y_left)

        if n_right == 0 or n_left == 0:
            return 0

        parent_impurity = self._impurity(y)
        right_impurity = self._impurity(y_right)
        left_impurity = self._impurity(y_left)

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        info_gain = parent_impurity - weighted_impurity

        return info_gain

    def _impurity(self, y):
        if len(y) == 0:
            return 0

        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)

    def _entropy(self, y):
        all_samples = len(y)
        entropy = 0

        counts = Counter(y).values()

        for count in counts:
            probability = count / all_samples
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    def _getlabel(self, y):
        return Counter(y).most_common(1)[0][0]


    def _gini_impurity(self, y):
        all_samples = len(y)
        impurity = 1

        counts = Counter(y).values()
        for count in counts:
            probability = count / all_samples
            impurity -= probability**2

        return impurity

    def _split(self, X, y, feature_inx, threshold):
        feature_column = X[:, feature_inx]
        right_inxs = feature_column >= threshold
        left_inxs = feature_column < threshold

        return X[right_inxs], X[left_inxs], y[right_inxs], y[left_inxs]

    def _best_split_params(self, X, y):
        _, n_features = X.shape

        # self.max_feature cannot be float
        if self.max_features == 'sqrt':
            max_features = math.sqrt(n_features)
        elif self.max_features == 'log2':
            max_features = np.log2(n_features)
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif self.max_features is None:
            max_features = n_features

        size = np.random.randint(1, max_features + 1)  # random size between 1 and max_features
        feature_indices = np.random.choice(X.shape[1], size=size, replace=False)

        if self.splitter == 'random':
            # Choose one index randomly but threshold is still the most optimal
            feature_indices = [np.random.choice(feature_indices)]

        best_info_gain = -np.inf
        best_threshold = None
        best_feature_inx = None

        for feature_inx in feature_indices:
            feature_column = X[:, feature_inx]
            thresholds = np.unique(feature_column)

            for threshold in thresholds:
                info_gain = self._info_gain(X, y, feature_inx, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_inx = feature_inx
                    best_threshold = threshold

        return best_feature_inx, best_threshold

    def predict(self, X):
        return [ self._predict(x, self._root_node) for x in X ]

    def _predict(self, x, node):
        if node.isleaf():
            return node.label

        if x[node.feature] >= node.threshold:
            return self._predict(x, node.right)

        return self._predict(x, node.left)

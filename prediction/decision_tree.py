import math
import numpy as np
from collections import Counter

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

class BaseDecisionTree:
    def __init__(
        self,
        splitter              = "best", # random
        max_depth             = None,
        min_samples_split     = 2,
        min_samples_leaf      = 1,
        max_features          = None,
        random_state          = None,
        max_leaf_nodes        = None,
        min_impurity_decrease = 0.0,
    ):
        self.splitter              = splitter
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.min_samples_leaf      = min_samples_leaf
        self.max_features          = max_features
        self.random_state          = random_state
        self.max_leaf_nodes        = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.root_node            = None
        self._leaf_count           = 0

        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        else:
            self.random_state = np.random.RandomState()


    def fit(self, X, y):
        self.root_node = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        node = Node()

        if self._impurity(y) == 0 or \
                (self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split or \
                (self.max_leaf_nodes is not None and self._leaf_count + 1 > self.max_leaf_nodes): # FIXME: doesn't work perfectly since leaf count can be one less than max leaf node
            node.label = self._getvalue(y)
            self._leaf_count += 1
            return node

        feature_inx, threshold, info_gain = self._best_split_params(X, y)
        if feature_inx == None or threshold == None or info_gain <= self.min_impurity_decrease:
            node.label = self._getvalue(y)
            self._leaf_count += 1
            return node

        X_right, X_left, y_right, y_left = self._split(X, y, feature_inx, threshold)

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node.label = self._getvalue(y)
            self._leaf_count += 1
            return node

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

    def _split(self, X, y, feature_inx, threshold):
        feature_column = X[:, feature_inx]
        right_inxs = feature_column >= threshold
        left_inxs = feature_column < threshold

        return X[right_inxs], X[left_inxs], y[right_inxs], y[left_inxs]

    def _best_split_params(self, X, y):
        _, n_features = X.shape

        # self.max_feature cannot be float
        if self.max_features == 'sqrt':
            max_features = int(math.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif self.max_features is None:
            max_features = n_features

        if max_features < n_features:
            size = self.random_state.randint(1, max_features + 1)  # random size between 1 and max_features
            feature_indices = self.random_state.choice(X.shape[1], size=size, replace=False)
        else:
            feature_indices = np.arange(X.shape[1])

        if self.splitter == 'random':
            # Choose one index randomly but threshold is still the most optimal
            feature_indices = [self.random_state.choice(feature_indices)]

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

        return best_feature_inx, best_threshold, best_info_gain

    def predict(self, X):
        return [ self._predict(x, self.root_node) for x in X ]

    def _predict(self, x, node):
        if node.isleaf():
            return node.label

        if x[node.feature] >= node.threshold:
            return self._predict(x, node.right)

        return self._predict(x, node.left)

    # Implemented in subclasses
    def _impurity(self, y):
        raise NotImplementedError("Use Classifier or Regressor")

    def _getvalue(self, y):
        raise NotImplementedError("Use Classifier or Regressor")

class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion = "gini", class_weight = None, **kwargs):
        self.criterion = criterion
        self.class_weight = class_weight
        super().__init__(**kwargs)

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

    def _getvalue(self, y):
        return Counter(y).most_common(1)[0][0]

    def _gini_impurity(self, y):
        all_samples = len(y)
        impurity = 1

        counts = Counter(y).values()
        for count in counts:
            probability = count / all_samples
            impurity -= probability**2

        return impurity

class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion = "mse", class_weight = None, **kwargs):
        self.criterion = criterion
        self.class_weight = class_weight
        super().__init__(**kwargs)

    def _impurity(self, y):
        if len(y) == 0:
            return 0

        if self.criterion == 'mse':
            return self._mse(y)
        if self.criterion == 'mae':
            return self._mae(y)

    def _mse(self, y):
        average = np.mean(y)
        mse = np.mean((y - average)**2)
        return mse

    def _mae(self, y):
        median = np.median(y)
        mae = np.mean(np.abs(y - median))
        return mae

    def _getvalue(self, y):
        return np.mean(y)

def print_tree(node, feature_names=None, indent=""):
    if node.isleaf():
        print(f"{indent}Predict: {node.label}")
        return

    feature_name = f"X[{node.feature}]" if feature_names is None else feature_names[node.feature]

    print(f"{indent}if {feature_name} < {node.threshold:.4f}:")
    print_tree(node.left, feature_names, indent + "  ")
    print(f"{indent}else:")
    print_tree(node.right, feature_names, indent + "  ")

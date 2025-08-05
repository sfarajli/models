class Node:
    def __init__(self, right, left, label=None):
        self.right = right
        self.left = left
        self.label = None

    def isleaf(self):
        if self.label:
            return True

        return False

class DecisionTree:
    def _gini():
        pass

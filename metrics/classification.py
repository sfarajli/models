import numpy as np

def _get_component(component, y_true, y_pred, pos_label):
    if component == 'tp':
        result = np.sum((y_true == pos_label) & (y_pred == pos_label))
    elif component == 'fp':
        result = np.sum((y_true != pos_label) & (y_pred == pos_label))
    elif component == 'tn':
        result = np.sum((y_true != pos_label) & (y_pred != pos_label))
    elif component == 'fn':
        result = np.sum((y_true == pos_label) & (y_pred != pos_label))
    else:
        raise ValueError(f"Unknown component: {component}")

    return result

def accuracy_score(y_true, y_pred):
    result = np.sum((y_true == y_pred)) / len(y_true)
    return result

def precision_score(y_true, y_pred, pos_label):
    tp = _get_component('tp', y_true, y_pred, pos_label)
    fp = _get_component('fp', y_true, y_pred, pos_label)

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)

def recall_score(y_true, y_pred, pos_label):
    tp = _get_component('tp', y_true, y_pred, pos_label)
    fn = _get_component('fn', y_true, y_pred, pos_label)

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)

def f1_score(y_true, y_pred, pos_label):
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred, pos_label):
    tn = _get_component('tn', y_true, y_pred, pos_label)
    fp = _get_component('fp', y_true, y_pred, pos_label)
    fn = _get_component('fn', y_true, y_pred, pos_label)
    tp = _get_component('tp', y_true, y_pred, pos_label)

    return np.array([[tn, fp], [fn, tp]])

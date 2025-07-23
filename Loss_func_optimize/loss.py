import numpy as np


def BCEWithLogitsLoss(logits, targets):
    logits = np.array(logits)
    targets = np.array(targets)
    loss = np.log1p(np.exp(-np.abs(logits))) + np.maximum(0, logits) - logits * targets
    return np.mean(loss)

def MSE(y,y_predict):
    return np.mean(y - y_predict) ** 2

def BCE(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    return -np.mean(loss)

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
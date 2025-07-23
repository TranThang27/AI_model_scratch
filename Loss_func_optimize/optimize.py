import numpy as np


def gradient_descent(y,X,epochs,lr):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        y_pred = np.dot(X,w) + b

        dloss_dw = (-2/n_samples) * np.dot (X.T ,(y-y_pred))
        dloss_db = (-2/n_samples) * np.sum(y-y_pred)

        w = w - lr * dloss_dw
        b = b - lr * dloss_db

    return w, b

def SGD(X, y, epochs, lr):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]
            y_pred = np.dot(xi, w) + b
            error = yi - y_pred

            dw = -2 * xi * error
            db = -2 * error

            w -= lr * dw
            b -= lr * db

    return w, b


def Momentum(X,y,epochs, lr , beta = 0.9):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    mt1 = np.zeros(n_features)
    mt2 = 0.0
    for epoch in range(epochs):
        y_pred = np.dot(X, w) + b

        dloss_dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
        dloss_db = (-2 / n_samples) * np.sum(y - y_pred)

        mt1 = beta * mt1 + (1-beta) * dloss_dw
        mt2 = beta * mt2 + (1 - beta) * dloss_db

        w = w - lr * mt1
        b = b - lr * mt2
    return w, b

def RMSprop(X, y, epochs, lr, beta=0.9, epsilon=1e-8):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    Eg2_w = np.zeros(n_features)
    Eg2_b = 0.0

    for epoch in range(epochs):
        y_pred = np.dot(X, w) + b
        error = y - y_pred

        grad_w = (-2 / n_samples) * np.dot(X.T, error)
        grad_b = (-2 / n_samples) * np.sum(error)

        Eg2_w = beta * Eg2_w + (1 - beta) * (grad_w ** 2)
        Eg2_b = beta * Eg2_b + (1 - beta) * (grad_b ** 2)

        w -= lr * grad_w / (np.sqrt(Eg2_w) + epsilon)
        b -= lr * grad_b / (np.sqrt(Eg2_b) + epsilon)

    return w, b


def Adam(X,y,epochs, lr , beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    mt1 = np.zeros(n_features)
    mt2 = 0.0
    vt1 =  np.zeros(n_features)
    vt2 = 0.0
    for t in range(1, epochs + 1):
        y_pred = np.dot(X , w) + b
        error = y - y_pred

        dw = (-2 / n_samples) * np.dot(X.T, error)
        db = (-2 / n_samples) * np.sum(error)

        mt1 = beta1 * mt1 + (1 - beta1) * dw
        mt2 = beta1 * mt2 + (1 - beta1) * db

        vt1 = beta2 * vt1 + (1-beta2) * (dw ** 2)
        vt2 = beta2 * vt2 + (1-beta2) * (db ** 2)

        mt_w_hat = mt1 / (1 - beta1 ** t)
        mt_b_hat = mt2 / (1 - beta1 ** t)
        vt_w_hat = vt1 / (1 - beta2 ** t)
        vt_b_hat = vt2 / (1 - beta2 ** t)

        w -= lr * mt_w_hat / (np.sqrt(vt_w_hat) + epsilon)
        b -= lr * mt_b_hat / (np.sqrt(vt_b_hat) + epsilon)
    return w, b


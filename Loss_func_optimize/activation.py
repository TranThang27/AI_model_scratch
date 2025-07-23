import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
def Relu(x):
    return np.maximum(0, x)

def LeakyRelu(x):
    return np.maximum(0.1*x, x)

def Tanh(x):
    return np.tanh(x)




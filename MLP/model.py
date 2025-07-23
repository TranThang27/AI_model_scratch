#Input: 3 neurons
#Hidden Layer 1: 4 neurons (ReLU)
#Hidden Layer 2: 4 neurons (ReLU)
#Output Layer:  2 neurons (Softmax)

import numpy as np
from Loss_func_optimize import activation

def relu_derivative(x):
    return (x > 0).astype(float)

class MLP:
    def __init__(self):
        self.w = [np.random.randn(3, 4) * np.sqrt(2 / 3),
            np.random.randn(4, 4) * np.sqrt(2 / 4),
            np.random.randn(4, 3) * np.sqrt(2 / 4)
                ]

        self.b = [ np.zeros((1, 4)),
            np.zeros((1, 4)),
            np.zeros((1, 3))
            ]

    def forward(self, X):
        self.a = [X]
        self.z = []

        #hidden layer 1
        z1 = np.dot(self.a[-1] , self.w[0]) + self.b[0]
        self.z.append(z1)
        a1 = activation.Relu(z1)
        self.a.append(a1)

        #hidden layer 2
        z2 = np.dot(self.a[-1], self.w[1]) + self.b[1]
        self.z.append(z2)
        a2 = activation.Relu(z2)
        self.a.append(a2)

        #ouput
        z3 = np.dot(self.a[-1], self.w[2]) + self.b[2]
        self.z.append(z3)
        a3 = activation.softmax(z3)
        self.a.append(a3)

        return a3

    def backprop(self, y_true, lr):
        m = y_true.shape[0]
        dz = self.a[-1] - y_true

        for i in reversed(range(3)):
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            self.w[i] -= lr * dw
            self.b[i] -= lr * db

            if i != 0:
                da = np.dot(dz, self.w[i].T)
                dz = da * relu_derivative(self.z[i - 1])





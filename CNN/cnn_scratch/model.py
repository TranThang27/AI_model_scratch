import numpy as np


def conv2d(img, kernels, stride=1, padding=1):
    num_kernels, kh, kw = kernels.shape
    h, w = img.shape
    img_padded = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    output = np.zeros((num_kernels, out_h, out_w))

    for k in range(num_kernels):
        kernel = kernels[k]
        for i in range(out_h):
            for j in range(out_w):
                region = img_padded[i * stride:i * stride + kh, j * stride:j * stride + kw]
                output[k, i, j] = np.sum(region * kernel)

    return output
def max_pooling(input, size=2, stride=2):
    C, H, W = input.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    output = np.zeros((C, out_h, out_w))

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = input[c, i*stride:i*stride+size, j*stride:j*stride+size]
                output[c, i, j] = np.max(region)

    return output


def flatten(x):
    return x.flatten()

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def he_init(shape):
    fan_in = shape[1] * shape[2] * shape[3] if len(shape) == 4 else shape[1]
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)
from model import conv2d, max_pooling, dense, flatten, he_init
from Loss_func_optimize import activation, loss
import numpy as np
import struct
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
        return images / 255.0  # normalize

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def conv2d_backward(grad_output, input_img, kernel, stride=1, padding=1):
    num_kernels, kh, kw = kernel.shape
    h, w = input_img.shape

    img_padded = np.pad(input_img, ((padding, padding), (padding, padding)), mode='constant')
    grad_kernel = np.zeros_like(kernel)
    grad_input = np.zeros_like(img_padded)

    out_h, out_w = grad_output.shape[1], grad_output.shape[2]

    for k in range(num_kernels):
        for i in range(out_h):
            for j in range(out_w):
                region = img_padded[i * stride:i * stride + kh, j * stride:j * stride + kw]
                grad_kernel[k] += grad_output[k, i, j] * region
                grad_input[i * stride:i * stride + kh, j * stride:j * stride + kw] += grad_output[k, i, j] * kernel[k]

    if padding > 0:
        grad_input = grad_input[padding:-padding, padding:-padding]
    else:
        grad_input = grad_input

    return grad_kernel, grad_input

def max_pooling_backward(grad_output, input_tensor, size=2, stride=2):
    C, H, W = input_tensor.shape
    grad_input = np.zeros_like(input_tensor)

    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = input_tensor[c, i * stride:i * stride + size, j * stride:j * stride + size]
                max_pos = np.unravel_index(np.argmax(region), region.shape)
                grad_input[c, i * stride + max_pos[0], j * stride + max_pos[1]] += grad_output[c, i, j]

    return grad_input

train_images = load_mnist_images('D:/track/pythonProject/CNN/data/MNIST/raw/train-images-idx3-ubyte')
train_labels = load_mnist_labels('D:/track/pythonProject/CNN/data/MNIST/raw/train-labels-idx1-ubyte')

batch_size = 32
num_samples = len(train_images)

# Fixed pooling parameters
conv_output_size = 28
pool_output_size = (conv_output_size - 2) // 2 + 1
dense_input_dim = pool_output_size * pool_output_size * 8

conv_kernel = he_init((8, 3, 3))
dense_weights = he_init((dense_input_dim, 10))
dense_bias = np.zeros((10,))
def forward(img, conv_kernel, dense_weights, dense_bias):
    conv_out = conv2d(img, conv_kernel, stride=1, padding=1)
    relu_out = activation.Relu(conv_out)
    pool_out = max_pooling(relu_out, size=2, stride=2)
    x_flat = flatten(pool_out)
    logits = dense(x_flat, dense_weights, dense_bias)
    probs = activation.softmax(logits)

    return probs, x_flat, logits, pool_out, relu_out, conv_out


lr = 0.01
epochs = 1

for epoch in range(epochs):
    total_loss = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        imgs_batch = train_images[batch_idx]
        labels_batch = train_labels[batch_idx]

        batch_loss = 0
        batch_grad_w = np.zeros_like(dense_weights)
        batch_grad_b = np.zeros_like(dense_bias)
        batch_grad_conv = np.zeros_like(conv_kernel)

        for j in range(len(imgs_batch)):
            img_np = imgs_batch[j]  # (28, 28)
            label_val = labels_batch[j]

            probs, x_flat, logits, pool_out, relu_out, conv_out = forward(img_np, conv_kernel, dense_weights,
                                                                          dense_bias)
            y_true = np.zeros(10)
            y_true[label_val] = 1
            loss_val = loss.cross_entropy_loss(y_true, probs)
            batch_loss += loss_val
            total_loss += loss_val

            grad_logits = probs.copy()
            grad_logits[label_val] -= 1
            grad_w = np.outer(x_flat, grad_logits)
            grad_b = grad_logits

            batch_grad_w += grad_w
            batch_grad_b += grad_b

            grad_x_flat = np.dot(dense_weights, grad_logits)
            grad_pool = grad_x_flat.reshape(pool_out.shape)
            grad_relu = max_pooling_backward(grad_pool, relu_out, size=2, stride=2)
            grad_conv = grad_relu.copy()
            grad_conv[conv_out <= 0] = 0
            grad_kernel, _ = conv2d_backward(grad_conv, img_np, conv_kernel, stride=1, padding=1)
            batch_grad_conv += grad_kernel

        dense_weights -= lr * batch_grad_w / len(imgs_batch)
        dense_bias -= lr * batch_grad_b / len(imgs_batch)
        conv_kernel -= lr * batch_grad_conv / len(imgs_batch)

        if i % 10 == 0:
            avg_batch_loss = batch_loss / len(imgs_batch)
            print(f"[Epoch {epoch + 1}] Batch {i // batch_size}, Avg Loss: {avg_batch_loss:.4f}")

    print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / num_samples:.4f}")

def save_model_numpy(conv_kernel, dense_weights, dense_bias, filepath):
    model_params = {
        'conv_kernel': conv_kernel,
        'dense_weights': dense_weights,
        'dense_bias': dense_bias
    }
    np.savez(filepath, **model_params)
    print(f"Model saved to {filepath}.npz")



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import MLP
import numpy as np
from Loss_func_optimize import loss
np.random.seed(42)
data = load_iris()
X = data.data[:, :3]  # Lấy 3 đặc trưng đầu tiên
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y_onehot = np.eye(3)[y]
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

model = MLP()

epochs = 1000
lr = 0.45

for i in range(epochs):
    y_pred = model.forward(X_train)

    loss1 = loss.cross_entropy_loss(y_train,y_pred)
    model.backprop(y_train, lr)

    if i % 100 == 0 :
        print(f"Epoch:{i} , loss {loss1:.4f} ")

y_test_pred = model.forward(X_test)
y_pred_class = np.argmax(y_test_pred, axis=1)
y_true_class = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred_class == y_true_class)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
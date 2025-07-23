import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

np.random.seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) , (0.5,))
])


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear( 32*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,10),
        )

    def forward(self, x):
        return self.net(x)

model = CNN()

lr = 0.01
citeration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr)

epochs = 10

for i in range (epochs):
    model.train()
    current_loss = 0.0
    total =0.0
    correct = 0.0
    for inputs, labels in trainloader:
        output = model(inputs)
        loss = citeration(output, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        current_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = correct/total
    print(f"Epoch: {i}, Loss: {current_loss:.4f}, Test Accuracy: {acc:.4f}")


torch.save(model.state_dict(), 'mnist_cnn_torch.pth')


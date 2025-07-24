import torch
import torch.nn as nn

chars = 'helo'
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

input_seq = 'hell'
target_seq = 'ello'

x = torch.tensor([char2idx[c] for c in input_seq], dtype=torch.long).unsqueeze(0)
y = torch.tensor([char2idx[c] for c in target_seq], dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out.squeeze(0))
        return out

vocab_size = len(chars)
embedding_dim = 10
hidden_dim = 20

model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 50

for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        pred = torch.argmax(output, dim=1)
        pred_str = ''.join([idx2char[i.item()] for i in pred])
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Predicted: {pred_str}")

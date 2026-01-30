"""
We are building a lightweight code completion model (similar to a tiny GitHub Copilot). We have tokenized source code data where integers represent specific syntax tokens (e.g., `def`, `return`, `int`, etc.).

Your task is to implement the training pipeline for a 'Next Token Prediction' task. We have provided a synthetic dataset class. You need to implement a simple model and the training loop.

**Specific Requirements:**

1.  **Data Handling:** You must treat this as a Causal Language Modeling (CLM) task. The model predicts the next token.

2.  **The Model and training Loop:** Implement the model. the forward pass, loss calculation, backward pass, and parameter update.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SyntheticCodeDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=100):
        # Random integers representing code tokens
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


vocab_size = 100
dataset = SyntheticCodeDataset(vocab_size=vocab_size)

# Create dataloader with batch size of 32
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class TinyCodeModel(nn.Module):
    """
    A simple feed-forward network for token prediction.
    Architecture: Embedding -> Linear -> ReLU -> Linear
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()

        # Define the layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)  # Output logits over vocab

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)       # (batch_size, seq_len, embed_dim)
        x = self.fc1(x)             # (batch_size, seq_len, hidden_dim)
        x = self.relu(x)
        x = self.fc2(x)             # (batch_size, seq_len, vocab_size)
        return x

def train_code_completion_model(model, dataloader, epochs=3, learning_rate=0.001):
    """
    Implement the training loop for Next Token Prediction.
    """

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set the model to train mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, sequences in enumerate(dataloader):
            # Prepare Inputs and Targets for Next Token Prediction
            # Input: all tokens except the last one
            # Target: all tokens except the first one (shifted by 1)
            inputs = sequences[:, :-1]   # (batch_size, seq_len - 1)
            targets = sequences[:, 1:]   # (batch_size, seq_len - 1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)  # (batch_size, seq_len - 1, vocab_size)

            # Reshape for CrossEntropyLoss: (N, C) and (N,)
            # logits: (batch_size * (seq_len - 1), vocab_size)
            # targets: (batch_size * (seq_len - 1),)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


my_model = TinyCodeModel(vocab_size)
train_code_completion_model(my_model, data_loader)

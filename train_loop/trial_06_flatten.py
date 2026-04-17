"""
We are building a lightweight model 

For simplicity, 
- we use a feed-forward model only.
- the model performs next token prediction, conditioned on only the current token.

You need to implement a simple model and the training loop.
Fill in the TODO sections.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=100):
        # Random integers representing code tokens
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


vocab_size = 100
dataset = SyntheticDataset(vocab_size=vocab_size)

# TODO: Create dataloader with batch size of 32
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class TinyModel(nn.Module):
    """
    A simple feed-forward network for token prediction.
    Architecture: Embedding -> Linear -> ReLU -> Linear
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()

        # TODO: Define the layers.
        self.emedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        # TODO: Implement the forward pass
        #x shape(batch_size * seq-1,) [5, 8, 2, 9]            MODIFIED
        # x (bsz, seq-1) -> (bsz * (seq-1))

        x = self.emedding(x)  #(batchsize* seq-1, embed_dim)
        x = self.fc1(x)       #(batchsize* seq-1, hidden_dim)
        x = self.relu(x)      #(batchsize* seq-1, hidden_dim)
        x = self.fc2(x)
        return x

def train_model(model, dataloader, epochs=1, learning_rate=0.001):
    """
    Implement the training loop for Next Token Prediction.
    """

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set the model to train mode
    model.train()

    # TODOs:
    # Prepare Inputs and Targets for Next Token Prediction
    # Complete the training loop

    for epoch in range(epochs):
        total_epoch_loss = 0

        for batch_idx, sequences in enumerate(dataloader):
            # TODO
            inputs = sequences[:, :-1]    #(batch_size, seq-1)
            targets = sequences[:, 1:]    #(batch_size, seq-1)
            optimizer.zero_grad()
            inputs = inputs.reshape(-1)    #(batch_size*seq-1)
            targets = targets.reshape(-1)    #(batch_size*seq-1)
            print(inputs.size(), targets.size())
            outputs = model(inputs)    #(batch_size*seq-1, vocab_size)
            print(outputs.size())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()


        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")


my_model = TinyModel(vocab_size)
train_model(my_model, data_loader)
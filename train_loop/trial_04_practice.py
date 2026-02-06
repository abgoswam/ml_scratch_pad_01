"""
We are building a lightweight model (e.g. Amazon Rufus).

Your task is to implement the training pipeline for a 'Next Token Prediction' task. We have provided a synthetic dataset class. 
You need to implement a simple model and the training loop.

**Specific Requirements:**

1.  **Data Handling:** You must treat this as a Causal Language Modeling (CLM) task. The model predicts the next token.
2.  **The Model and training Loop:** Implement the model. the forward pass, loss calculation, backward pass, and parameter update.

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

# TOD: Create dataloader with batch size of 32
data_loader = pass

class TinyModel(nn.Module):
    """
    A simple feed-forward network for token prediction.
    Architecture: Embedding -> Linear -> ReLU -> Linear
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()

        # TOD:: Define the layers.
        pass

    def forward(self, x):
        # TODO: Implement the forward pass
        pass

def train_model(model, dataloader, epochs=3, learning_rate=0.001):
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
            pass

        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")


my_model = TinyModel(vocab_size)
train_model(my_model, data_loader)

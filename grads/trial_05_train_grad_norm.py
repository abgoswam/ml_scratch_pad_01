# Let's create a small synthetic dataset and a tiny neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_data(class0, class1, title="Dataset Visualization"):
    """Visualize the 2D dataset with separate class0 and class1 tensors"""
    plt.figure(figsize=(8, 6))
    
    # Convert to numpy for plotting
    class0_np = class0.detach().numpy()
    class1_np = class1.detach().numpy()
    
    # Plot class 0 points in blue
    plt.scatter(class0_np[:, 0], class0_np[:, 1], 
                c='blue', label='Class 0', alpha=0.7, s=50)
    
    # Plot class 1 points in red
    plt.scatter(class1_np[:, 0], class1_np[:, 1], 
                c='red', label='Class 1', alpha=0.7, s=50)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(title)
    plt.legend()
    
    # Add center points for reference
    plt.axvline(x=-2.0, color='blue', linestyle='--', alpha=0.5, label='Class 0 center')
    plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='Class 1 center')
    
    plt.tight_layout()
    plt.show()

def calculate_grad_norm(model):
    """Calculate the L2 norm of gradients across all model parameters"""
    total_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    return total_norm ** 0.5

def plot_training_metrics(losses, grad_norms):
    """Plot loss and gradient norm over training steps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot gradient norm
    ax2.plot(grad_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Synthetic dataset: simple 2D points in two classes
    torch.manual_seed(0)
    N = 20
    class0 = torch.randn(N, 2) + torch.tensor([-2.0, 0.0])
    class1 = torch.randn(N, 2) + torch.tensor([2.0, 0.0])
    show_data(class0, class1)

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)])

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Simple model: 2 -> 2 (two class logits)
    model = nn.Linear(2, 2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def get_logits_and_loss():
        logits = model(X)           # raw scores
        loss = F.cross_entropy(logits, y)
        return logits.detach(), loss

    print("Before training:")
    logits, loss = get_logits_and_loss()
    print("Logits (first 5 rows):\n", logits[:5])
    print("Loss:", loss.item())

    # Lists to store metrics for plotting
    losses = []
    grad_norms = []

    # Train for a few steps
    for step in range(20):
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        # Calculate gradient norm before optimizer step
        grad_norm = calculate_grad_norm(model)
        
        # Store metrics
        losses.append(loss.item())
        grad_norms.append(grad_norm)
        
        optimizer.step()

        if step % 5 == 0:
            print(f"\nStep {step}")
            print("Loss:", loss.item())
            print("Gradient norm:", grad_norm)
            print("Sample logits:", logits[:1].detach())

    print("\nAfter training:")
    logits, loss = get_logits_and_loss()
    print("Logits (first 5 rows):\n", logits[:5])
    print("Loss:", loss.item())
    
    # Plot training metrics
    plot_training_metrics(losses, grad_norms)
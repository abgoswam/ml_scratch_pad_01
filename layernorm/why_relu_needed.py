import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(123)

def demonstrate_nonlinearity():
    """Demonstrate why nonlinearity is crucial for neural networks"""
    
    # Create a simple 2D dataset that's not linearly separable
    n_samples = 100
    
    # XOR-like problem: points should be classified based on which quadrant they're in
    x = torch.randn(n_samples, 2) * 2
    # XOR pattern: positive if both coords have same sign, negative otherwise
    y = ((x[:, 0] > 0) == (x[:, 1] > 0)).float()
    
    print("Dataset characteristics:")
    print(f"Input shape: {x.shape}")
    print(f"Class 0 samples: {(y == 0).sum()}")
    print(f"Class 1 samples: {(y == 1).sum()}")
    
    # Plot the dataset
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(x[:, 0], x[:, 1], c=colors, alpha=0.7)
    plt.title('XOR-like Dataset\n(Not Linearly Separable)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    
    # Try to solve with LINEAR model only (no activation)
    linear_model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.01)
    
    # Train linear model
    losses_linear = []
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = linear_model(x).squeeze()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses_linear.append(loss.item())
    
    # Test linear model
    with torch.no_grad():
        pred_linear = linear_model(x).squeeze()
        accuracy_linear = ((pred_linear > 0.5) == y).float().mean()
    
    plt.subplot(1, 3, 2)
    pred_colors = ['red' if p < 0.5 else 'blue' for p in pred_linear]
    plt.scatter(x[:, 0], x[:, 1], c=pred_colors, alpha=0.7)
    plt.title(f'Linear Model Predictions\nAccuracy: {accuracy_linear:.2f}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    
    # Now try with ReLU nonlinearity
    relu_model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    
    optimizer_relu = torch.optim.Adam(relu_model.parameters(), lr=0.01)
    
    # Train ReLU model
    losses_relu = []
    for epoch in range(1000):
        optimizer_relu.zero_grad()
        pred = relu_model(x).squeeze()
        loss = criterion(pred, y)
        loss.backward()
        optimizer_relu.step()
        losses_relu.append(loss.item())
    
    # Test ReLU model
    with torch.no_grad():
        pred_relu = relu_model(x).squeeze()
        accuracy_relu = ((pred_relu > 0.5) == y).float().mean()
    
    plt.subplot(1, 3, 3)
    pred_colors_relu = ['red' if p < 0.5 else 'blue' for p in pred_relu]
    plt.scatter(x[:, 0], x[:, 1], c=pred_colors_relu, alpha=0.7)
    plt.title(f'ReLU Model Predictions\nAccuracy: {accuracy_relu:.2f}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults:")
    print(f"Linear model accuracy: {accuracy_linear:.3f}")
    print(f"ReLU model accuracy: {accuracy_relu:.3f}")
    
    return losses_linear, losses_relu

def demonstrate_universal_approximation():
    """Show how ReLU enables universal approximation"""
    
    # Create a complex function to approximate
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_true = torch.sin(x) * torch.exp(-x**2/2)  # Complex non-linear function
    
    plt.figure(figsize=(15, 5))
    
    # Plot target function
    plt.subplot(1, 3, 1)
    plt.plot(x.numpy(), y_true.numpy(), 'k-', linewidth=2, label='Target Function')
    plt.title('Target: sin(x) * exp(-x²/2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Try with linear model
    linear_approx = nn.Sequential(
        nn.Linear(1, 50),
        nn.Linear(50, 50),
        nn.Linear(50, 1)
    )
    
    optimizer = torch.optim.Adam(linear_approx.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(2000):
        optimizer.zero_grad()
        pred = linear_approx(x)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        y_linear = linear_approx(x)
    
    plt.subplot(1, 3, 2)
    plt.plot(x.numpy(), y_true.numpy(), 'k-', linewidth=2, label='Target')
    plt.plot(x.numpy(), y_linear.numpy(), 'r--', linewidth=2, label='Linear Model')
    plt.title('Linear Model Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Try with ReLU model
    relu_approx = nn.Sequential(
        nn.Linear(1, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    optimizer_relu = torch.optim.Adam(relu_approx.parameters(), lr=0.01)
    
    for epoch in range(2000):
        optimizer_relu.zero_grad()
        pred = relu_approx(x)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer_relu.step()
    
    with torch.no_grad():
        y_relu = relu_approx(x)
        mse_linear = nn.MSELoss()(y_linear, y_true)
        mse_relu = nn.MSELoss()(y_relu, y_true)
    
    plt.subplot(1, 3, 3)
    plt.plot(x.numpy(), y_true.numpy(), 'k-', linewidth=2, label='Target')
    plt.plot(x.numpy(), y_relu.numpy(), 'b--', linewidth=2, label='ReLU Model')
    plt.title('ReLU Model Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Function Approximation Results:")
    print(f"Linear model MSE: {mse_linear:.6f}")
    print(f"ReLU model MSE: {mse_relu:.6f}")
    print(f"ReLU improvement: {(mse_linear/mse_relu):.1f}x better")

def demonstrate_gradient_flow():
    """Show gradient flow differences"""
    
    print("\n" + "="*60)
    print("GRADIENT FLOW DEMONSTRATION")
    print("="*60)
    
    # Create a simple deep network
    x = torch.randn(1, 10, requires_grad=True)
    
    # Linear model (without activation)
    linear_layers = []
    for i in range(5):
        linear_layers.append(nn.Linear(10, 10))
    linear_model = nn.Sequential(*linear_layers)
    
    # ReLU model
    relu_layers = []
    for i in range(5):
        relu_layers.extend([nn.Linear(10, 10), nn.ReLU()])
    relu_model = nn.Sequential(*relu_layers[:-1])  # Remove last ReLU
    
    # Forward pass and compute gradients
    y_linear = linear_model(x).sum()
    y_relu = relu_model(x).sum()
    
    y_linear.backward(retain_graph=True)
    linear_grads = [p.grad.norm().item() if p.grad is not None else 0 
                   for p in linear_model.parameters()]
    
    # Clear gradients
    for p in linear_model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    y_relu.backward()
    relu_grads = [p.grad.norm().item() if p.grad is not None else 0 
                 for p in relu_model.parameters()]
    
    # Get weight gradients only (skip bias gradients for cleaner display)
    linear_weight_grads = linear_grads[::2]  # Every other parameter (weights)
    relu_weight_grads = relu_grads[::2]      # Every other parameter (weights)
    
    print(f"Gradient norms in deep networks:")
    print(f"Linear model gradient norms: {linear_weight_grads}")
    print(f"ReLU model gradient norms: {relu_weight_grads}")
    
    # Show the mathematical reason
    print(f"\nKey insight:")
    print(f"- Linear: All layers just matrix multiplication")
    print(f"- ReLU: Introduces nonlinearity, prevents gradient vanishing/exploding")

if __name__ == "__main__":
    print("Why ReLU is Needed: Comprehensive Demonstration")
    print("="*60)
    
    print("\n1. NONLINEARITY: Solving Non-Linear Problems")
    print("-" * 50)
    losses_linear, losses_relu = demonstrate_nonlinearity()
    
    print("\n2. UNIVERSAL APPROXIMATION: Complex Function Learning")
    print("-" * 50)
    demonstrate_universal_approximation()
    
    print("\n3. GRADIENT FLOW: Training Deep Networks")
    print("-" * 50)
    demonstrate_gradient_flow()
    
    print("\n" + "="*60)
    print("SUMMARY: Why ReLU is Essential")
    print("="*60)
    print("""
1. ENABLES NONLINEARITY:
   - Without nonlinear activations, deep networks collapse to linear models
   - Can't solve problems like XOR, image recognition, etc.
   
2. UNIVERSAL APPROXIMATION:
   - ReLU networks can approximate any continuous function
   - Linear networks are severely limited in what they can learn
   
3. COMPUTATIONAL EFFICIENCY:
   - Simple: max(0, x)
   - Fast forward and backward pass
   - No saturation problems like sigmoid/tanh
   
4. GRADIENT FLOW:
   - No vanishing gradient problem (for positive values)
   - Enables training of very deep networks
   
5. SPARSITY:
   - Creates sparse representations (many zeros)
   - Can improve generalization and efficiency
    """)
    
    print("done")
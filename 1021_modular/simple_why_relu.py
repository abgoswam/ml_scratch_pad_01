import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

def simple_nonlinearity_demo():
    """Most important reason: Without ReLU, deep networks = single linear layer!"""
    
    print("CRITICAL INSIGHT: Why ReLU is Absolutely Essential")
    print("="*60)
    
    # Create input
    x = torch.randn(3, 4)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x}")
    
    print("\n" + "-"*40)
    print("CASE 1: Deep Network WITHOUT ReLU")
    print("-"*40)
    
    # Multiple linear layers (no activation)
    linear_net = nn.Sequential(
        nn.Linear(4, 8),
        nn.Linear(8, 6),
        nn.Linear(6, 2)
    )
    
    # This is mathematically equivalent to a single linear layer!
    output1 = linear_net(x)
    
    # Let's prove this by computing the equivalent single matrix
    W1 = linear_net[0].weight.data
    b1 = linear_net[0].bias.data
    W2 = linear_net[1].weight.data  
    b2 = linear_net[1].bias.data
    W3 = linear_net[2].weight.data
    b3 = linear_net[2].bias.data
    
    # Mathematical composition: W3(W2(W1*x + b1) + b2) + b3
    # = W3*W2*W1*x + W3*W2*b1 + W3*b2 + b3
    W_combined = W3 @ W2 @ W1  # Matrix multiplication combines
    b_combined = W3 @ W2 @ b1 + W3 @ b2 + b3
    
    # Single equivalent linear layer
    single_linear = nn.Linear(4, 2)
    single_linear.weight.data = W_combined
    single_linear.bias.data = b_combined
    
    output1_equivalent = single_linear(x)
    
    print(f"Deep linear network output:\n{output1}")
    print(f"Equivalent single linear layer output:\n{output1_equivalent}")
    print(f"Difference (should be ~0): {(output1 - output1_equivalent).abs().max().item():.10f}")
    
    print("\n" + "-"*40)
    print("CASE 2: Deep Network WITH ReLU")
    print("-"*40)
    
    # Same network but with ReLU
    relu_net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 6), 
        nn.ReLU(),
        nn.Linear(6, 2)
    )
    
    output2 = relu_net(x)
    print(f"Deep ReLU network output:\n{output2}")
    
    # Try to approximate with single linear layer
    best_single = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(best_single.parameters(), lr=0.1)
    
    # Train single layer to match ReLU network
    for _ in range(1000):
        optimizer.zero_grad()
        pred = best_single(x)
        loss = nn.MSELoss()(pred, output2.detach())
        loss.backward()
        optimizer.step()
    
    best_single_output = best_single(x)
    approximation_error = nn.MSELoss()(best_single_output, output2).item()
    
    print(f"Best single linear approximation:\n{best_single_output}")
    print(f"Approximation error: {approximation_error:.6f}")
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print("1. WITHOUT ReLU: Deep network = Single linear layer")
    print("   → Can only learn linear relationships")
    print("   → Completely equivalent mathematically")
    print()
    print("2. WITH ReLU: Deep network ≠ Single linear layer") 
    print("   → Can learn complex non-linear relationships")
    print("   → Each layer adds representational power")
    print()
    print("3. REAL WORLD: Most problems are non-linear")
    print("   → Image recognition, language, speech, etc.")
    print("   → Linear models fundamentally insufficient")
    
def show_linear_limitation():
    """Show what happens when we try to solve a simple non-linear problem"""
    
    print("\n" + "="*60)
    print("PRACTICAL EXAMPLE: Why Linear Models Fail")
    print("="*60)
    
    # Simple non-linear problem: learn y = x^2
    x = torch.linspace(-2, 2, 100).unsqueeze(1)
    y_true = x**2
    
    # Linear model
    linear_model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = linear_model(x)
        loss = nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()
    
    # ReLU model  
    relu_model = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(), 
        nn.Linear(20, 1)
    )
    
    optimizer_relu = torch.optim.Adam(relu_model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        optimizer_relu.zero_grad()
        y_pred = relu_model(x)
        loss = nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer_relu.step()
    
    with torch.no_grad():
        y_linear = linear_model(x)
        y_relu = relu_model(x)
        
        linear_error = nn.MSELoss()(y_linear, y_true).item()
        relu_error = nn.MSELoss()(y_relu, y_true).item()
    
    print(f"Target function: y = x²")
    print(f"Linear model error: {linear_error:.6f}")
    print(f"ReLU model error: {relu_error:.6f}")
    print(f"ReLU is {linear_error/relu_error:.1f}x better!")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_true.numpy(), 'k-', linewidth=3, label='True: y = x²')
    plt.plot(x.numpy(), y_linear.numpy(), 'r--', linewidth=2, label=f'Linear (MSE: {linear_error:.4f})')
    plt.plot(x.numpy(), y_relu.numpy(), 'b:', linewidth=2, label=f'ReLU (MSE: {relu_error:.4f})')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('Linear vs ReLU: Learning y = x²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    simple_nonlinearity_demo()
    show_linear_limitation()
    
    print("\n" + "="*60)
    print("FINAL ANSWER: Why ReLU is Needed")
    print("="*60)
    print("""
ReLU is needed because:

1. WITHOUT ReLU: 
   - Any deep network collapses to a single linear transformation
   - Can only draw straight lines/planes in high dimensions
   - Useless for real-world problems

2. WITH ReLU:
   - Each layer can learn different non-linear patterns  
   - Can approximate ANY continuous function
   - Solves real problems: vision, language, etc.

3. MATHEMATICAL FACT:
   - Linear ∘ Linear ∘ Linear = Linear (always!)
   - ReLU ∘ Linear ∘ ReLU ∘ Linear ≠ Linear (expressive!)

4. PRACTICAL IMPACT:
   - ReLU: "Let me learn curves, edges, patterns..."
   - No ReLU: "I can only fit a straight line... 😢"
    """)
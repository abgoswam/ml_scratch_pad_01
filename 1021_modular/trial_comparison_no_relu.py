import torch
torch.manual_seed(123)

import torch.nn as nn
import matplotlib.pyplot as plt

def plot_samples(x_before, x_after, title_prefix=""):
    # x shape (b, emb_size)
    # draw a histogram of all the different b rows.  plot each rown in a separate color.
    # display the mean and variance of rach row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Plot before (left subplot)
    for i in range(x_before.shape[0]):
        row = x_before[i].detach().numpy()
        mean_val = row.mean()
        var_val = row.var()
        color = colors[i % len(colors)]
        
        ax1.hist(row, alpha=0.6, color=color, bins=20, 
                label=f'Row {i}: μ={mean_val:.3f}, σ²={var_val:.3f}')
    
    ax1.legend()
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title_prefix}Before Layer Norm')
    
    # Plot after (right subplot)
    for i in range(x_after.shape[0]):
        row = x_after[i].detach().numpy()
        mean_val = row.mean()
        var_val = row.var()
        color = colors[i % len(colors)]
        
        ax2.hist(row, alpha=0.6, color=color, bins=20, 
                label=f'Row {i}: μ={mean_val:.3f}, σ²={var_val:.3f}')
    
    ax2.legend()
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title_prefix}After Layer Norm')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    batch_example = torch.randn(2, 5)   
    print(f"batch_example:\n{batch_example}")

    # WITHOUT ReLU
    print("\n" + "="*50)
    print("WITHOUT ReLU")
    print("="*50)
    
    layer_no_relu = nn.Sequential(
        nn.Linear(5, 128)
        # No ReLU here
    )
    
    out_no_relu = layer_no_relu(batch_example)
    mean_no_relu = out_no_relu.mean(dim=-1, keepdim=True)
    var_no_relu = out_no_relu.var(dim=-1, keepdim=True)
    
    print(f"out_no_relu stats:")
    print(f"Min: {out_no_relu.min():.4f}, Max: {out_no_relu.max():.4f}")
    print(f"Values around 0: {(torch.abs(out_no_relu) < 0.1).sum().item()}/{out_no_relu.numel()}")
    
    # working on layer norm.
    out_norm_no_relu = (out_no_relu - mean_no_relu) / torch.sqrt(var_no_relu)
    
    plot_samples(out_no_relu, out_norm_no_relu, "NO ReLU - ")
    
    # WITH ReLU
    print("\n" + "="*50)
    print("WITH ReLU")
    print("="*50)
    
    layer_with_relu = nn.Sequential(
        nn.Linear(5, 128), 
        nn.ReLU()
    )
    
    out_with_relu = layer_with_relu(batch_example)
    mean_with_relu = out_with_relu.mean(dim=-1, keepdim=True)
    var_with_relu = out_with_relu.var(dim=-1, keepdim=True)
    
    print(f"out_with_relu stats:")
    print(f"Min: {out_with_relu.min():.4f}, Max: {out_with_relu.max():.4f}")
    print(f"Values equal to 0: {(out_with_relu == 0).sum().item()}/{out_with_relu.numel()}")
    print(f"Percentage of zeros: {(out_with_relu == 0).sum().item()/out_with_relu.numel()*100:.1f}%")
    
    # working on layer norm.
    out_norm_with_relu = (out_with_relu - mean_with_relu) / torch.sqrt(var_with_relu)
    
    plot_samples(out_with_relu, out_norm_with_relu, "WITH ReLU - ")
    
    print("done")
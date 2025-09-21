import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data():
    """Load and parse the results.log file"""
    # Read the file
    file_path = Path(__file__).parent / "results.log"
    
    # Read the data
    df = pd.read_csv(file_path, sep='\t')
    
    return df

def plot_training_metrics(df):
    """Plot basic training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Training Loss
    axes[0, 0].plot(df['Step'], df['Training Loss'], 'b-', marker='o')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Reward with error bars
    axes[0, 1].errorbar(df['Step'], df['reward'], yerr=df['reward_std'], 
                       capsize=3, marker='o', linestyle='-')
    axes[0, 1].set_title('Reward (with std)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # KL Divergence
    axes[1, 0].plot(df['Step'], df['kl'], 'r-', marker='s')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('KL')
    axes[1, 0].grid(True)
    
    # Clipped Ratio
    axes[1, 1].plot(df['Step'], df['completions / clipped_ratio'], 'g-', marker='^')
    axes[1, 1].set_title('Clipped Ratio')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_completion_lengths(df):
    """Plot completion length statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Completion Length Statistics', fontsize=16)
    
    # Mean, min, max lengths
    axes[0, 0].plot(df['Step'], df['completions / mean_length'], 'b-', marker='o', label='Mean')
    axes[0, 0].plot(df['Step'], df['completions / min_length'], 'g-', marker='s', label='Min')
    axes[0, 0].plot(df['Step'], df['completions / max_length'], 'r-', marker='^', label='Max')
    axes[0, 0].set_title('Completion Lengths')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Length')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Terminated lengths (where applicable)
    # Only plot when terminated lengths are non-zero
    mask = df['completions / mean_terminated_length'] > 0
    if mask.sum() > 0:
        axes[0, 1].plot(df.loc[mask, 'Step'], df.loc[mask, 'completions / mean_terminated_length'], 
                       'b-', marker='o', label='Mean Terminated')
        axes[0, 1].plot(df.loc[mask, 'Step'], df.loc[mask, 'completions / min_terminated_length'], 
                       'g-', marker='s', label='Min Terminated')
        axes[0, 1].plot(df.loc[mask, 'Step'], df.loc[mask, 'completions / max_terminated_length'], 
                       'r-', marker='^', label='Max Terminated')
        axes[0, 1].set_title('Terminated Completion Lengths')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].text(0.5, 0.5, 'No terminated completions', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Terminated Completion Lengths')
    
    # Distribution visualization
    axes[1, 0].fill_between(df['Step'], df['completions / min_length'], 
                           df['completions / max_length'], alpha=0.3, color='lightblue', 
                           label='Length Range')
    axes[1, 0].plot(df['Step'], df['completions / mean_length'], 'b-', marker='o', label='Mean')
    axes[1, 0].set_title('Length Distribution')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Clipped ratio over time
    axes[1, 1].bar(df['Step'], df['completions / clipped_ratio'], width=0.8, alpha=0.7)
    axes[1, 1].set_title('Clipped Ratio per Step')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Clipped Ratio')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_reward_components(df):
    """Plot different reward component metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Components', fontsize=16)
    
    # Match format exactly
    axes[0, 0].errorbar(df['Step'], df['rewards / match_format_exactly / mean'], 
                       yerr=df['rewards / match_format_exactly / std'], 
                       capsize=3, marker='o', linestyle='-', color='blue')
    axes[0, 0].set_title('Match Format Exactly')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Match format approximately
    axes[0, 1].errorbar(df['Step'], df['rewards / match_format_approximately / mean'], 
                       yerr=df['rewards / match_format_approximately / std'], 
                       capsize=3, marker='s', linestyle='-', color='green')
    axes[0, 1].set_title('Match Format Approximately')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # Check answer
    axes[1, 0].errorbar(df['Step'], df['rewards / check_answer / mean'], 
                       yerr=df['rewards / check_answer / std'], 
                       capsize=3, marker='^', linestyle='-', color='red')
    axes[1, 0].set_title('Check Answer')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True)
    
    # Check numbers
    axes[1, 1].errorbar(df['Step'], df['rewards / check_numbers / mean'], 
                       yerr=df['rewards / check_numbers / std'], 
                       capsize=3, marker='d', linestyle='-', color='orange')
    axes[1, 1].set_title('Check Numbers')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_reward_comparison(df):
    """Plot all reward components together for comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Reward Components Comparison', fontsize=16)
    
    # Mean values
    ax1.plot(df['Step'], df['rewards / match_format_exactly / mean'], 
            'b-', marker='o', label='Match Format Exactly')
    ax1.plot(df['Step'], df['rewards / match_format_approximately / mean'], 
            'g-', marker='s', label='Match Format Approximately')
    ax1.plot(df['Step'], df['rewards / check_answer / mean'], 
            'r-', marker='^', label='Check Answer')
    ax1.plot(df['Step'], df['rewards / check_numbers / mean'], 
            'orange', marker='d', label='Check Numbers')
    ax1.plot(df['Step'], df['reward'], 'k--', marker='x', label='Total Reward', linewidth=2)
    
    ax1.set_title('Reward Component Means')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Standard deviations
    ax2.plot(df['Step'], df['rewards / match_format_exactly / std'], 
            'b-', marker='o', label='Match Format Exactly')
    ax2.plot(df['Step'], df['rewards / match_format_approximately / std'], 
            'g-', marker='s', label='Match Format Approximately')
    ax2.plot(df['Step'], df['rewards / check_answer / std'], 
            'r-', marker='^', label='Check Answer')
    ax2.plot(df['Step'], df['rewards / check_numbers / std'], 
            'orange', marker='d', label='Check Numbers')
    ax2.plot(df['Step'], df['reward_std'], 'k--', marker='x', label='Total Reward Std', linewidth=2)
    
    ax2.set_title('Reward Component Standard Deviations')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Create a correlation heatmap of key metrics"""
    # Select numerical columns for correlation
    numerical_cols = [
        'Training Loss', 'reward', 'reward_std', 
        'completions / mean_length', 'kl',
        'rewards / match_format_exactly / mean',
        'rewards / match_format_approximately / mean',
        'rewards / check_answer / mean',
        'rewards / check_numbers / mean'
    ]
    
    corr_data = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks and labels
    ax.set_xticks(range(len(numerical_cols)))
    ax.set_yticks(range(len(numerical_cols)))
    ax.set_xticklabels([col.replace('rewards / ', '').replace(' / mean', '') for col in numerical_cols], 
                      rotation=45, ha='right')
    ax.set_yticklabels([col.replace('rewards / ', '').replace(' / mean', '') for col in numerical_cols])
    
    # Add correlation values as text
    for i in range(len(numerical_cols)):
        for j in range(len(numerical_cols)):
            text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black" if abs(corr_data.iloc[i, j]) < 0.5 else "white")
    
    ax.set_title("Correlation Heatmap of Key Metrics")
    plt.tight_layout()
    return fig

def main():
    """Main function to generate all plots"""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} steps of training data")
    print(f"Columns: {list(df.columns)}")
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating plots...")
    
    # Generate all plots
    fig1 = plot_training_metrics(df)
    fig1.savefig(output_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
    print("✓ Training metrics plot saved")
    
    fig2 = plot_completion_lengths(df)
    fig2.savefig(output_dir / "completion_lengths.png", dpi=300, bbox_inches='tight')
    print("✓ Completion lengths plot saved")
    
    fig3 = plot_reward_components(df)
    fig3.savefig(output_dir / "reward_components.png", dpi=300, bbox_inches='tight')
    print("✓ Reward components plot saved")
    
    fig4 = plot_reward_comparison(df)
    fig4.savefig(output_dir / "reward_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Reward comparison plot saved")
    
    fig5 = plot_correlation_heatmap(df)
    fig5.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved")
    
    # Show basic statistics
    print("\n=== Data Summary ===")
    print(f"Training steps: {df['Step'].min()} to {df['Step'].max()}")
    print(f"Final training loss: {df['Training Loss'].iloc[-1]:.6f}")
    print(f"Final reward: {df['reward'].iloc[-1]:.3f} ± {df['reward_std'].iloc[-1]:.3f}")
    print(f"Final KL divergence: {df['kl'].iloc[-1]:.6f}")
    print(f"Average completion length: {df['completions / mean_length'].mean():.1f}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
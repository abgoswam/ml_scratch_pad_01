import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_data():
    """Load and parse the results2.log file"""
    file_path = Path(__file__).parent / "results2.log"
    df = pd.read_csv(file_path, sep='\t')
    return df

def analyze_reward_trends(df):
    """Analyze if rewards increase per iteration"""
    print("=== Reward Trend Analysis ===")
    
    # Overall reward trend
    initial_reward = df['reward'].iloc[0]
    final_reward = df['reward'].iloc[-1]
    max_reward = df['reward'].max()
    min_reward = df['reward'].min()
    
    print(f"Initial reward (Step 1): {initial_reward:.3f}")
    print(f"Final reward (Step 100): {final_reward:.3f}")
    print(f"Maximum reward: {max_reward:.3f} (Step {df.loc[df['reward'].idxmax(), 'Step']})")
    print(f"Minimum reward: {min_reward:.3f}")
    print(f"Overall change: {final_reward - initial_reward:.3f}")
    
    # Calculate rolling averages to see trends
    window_size = 10
    df['reward_rolling_mean'] = df['reward'].rolling(window=window_size, center=True).mean()
    
    # Check if there's an upward trend
    correlation_with_step = df['reward'].corr(df['Step'])
    print(f"Correlation between Step and Reward: {correlation_with_step:.3f}")
    
    if correlation_with_step > 0.1:
        print("✓ Positive trend: Rewards generally increase over iterations")
    elif correlation_with_step < -0.1:
        print("✗ Negative trend: Rewards generally decrease over iterations")
    else:
        print("~ Neutral trend: No clear increase/decrease pattern")
    
    # Analyze individual reward components
    print("\n=== Individual Reward Component Trends ===")
    reward_cols = ['rewards / match_format_exactly', 'rewards / match_format_approximately', 
                   'rewards / check_answer', 'rewards / check_numbers']
    
    for col in reward_cols:
        corr = df[col].corr(df['Step'])
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        print(f"{col.replace('rewards / ', ''):<30}: {corr:+.3f} correlation, {initial:+.3f} → {final:+.3f}")
    
    return df

def plot_reward_overview(df):
    """Plot comprehensive reward analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Reward Analysis - Results2.log (100 Steps)', fontsize=16, fontweight='bold')
    
    # Main reward with rolling average
    axes[0, 0].plot(df['Step'], df['reward'], 'b-', alpha=0.6, linewidth=1, label='Actual Reward')
    axes[0, 0].plot(df['Step'], df['reward_rolling_mean'], 'r-', linewidth=2, label='Rolling Mean (10 steps)')
    
    # Add trend line
    z = np.polyfit(df['Step'], df['reward'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['Step'], p(df['Step']), '--', color='green', 
                   label=f'Trend (slope={z[0]:.3f})', alpha=0.8)
    
    axes[0, 0].set_title('Overall Reward Progression')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[0, 1].hist(df['reward'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['reward'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["reward"].mean():.2f}')
    axes[0, 1].axvline(df['reward'].median(), color='green', linestyle='--', 
                      label=f'Median: {df["reward"].median():.2f}')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Reward Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward variance over time
    axes[1, 0].plot(df['Step'], df['reward_std'], 'purple', marker='o', markersize=2)
    axes[1, 0].set_title('Reward Standard Deviation')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative reward
    axes[1, 1].plot(df['Step'], df['reward'].cumsum(), 'orange', linewidth=2)
    axes[1, 1].set_title('Cumulative Reward')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_reward_components(df):
    """Plot individual reward components"""
    reward_cols = ['rewards / match_format_exactly', 'rewards / match_format_approximately', 
                   'rewards / check_answer', 'rewards / check_numbers']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Individual Reward Components Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (col, color) in enumerate(zip(reward_cols, colors)):
        row, col_idx = i // 2, i % 2
        ax = axes[row, col_idx]
        
        # Plot the component
        ax.plot(df['Step'], df[col], color=color, alpha=0.6, linewidth=1, label='Actual')
        
        # Add rolling average
        rolling_mean = df[col].rolling(window=10, center=True).mean()
        ax.plot(df['Step'], rolling_mean, color=color, linewidth=2, alpha=0.8, label='Rolling Mean')
        
        # Add trend line
        z = np.polyfit(df['Step'], df[col], 1)
        p = np.poly1d(z)
        ax.plot(df['Step'], p(df['Step']), '--', color='black', 
               label=f'Trend (slope={z[0]:.4f})', alpha=0.7)
        
        # Formatting
        title = col.replace('rewards / ', '').replace('_', ' ').title()
        ax.set_title(f'{title}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation annotation
        corr = df[col].corr(df['Step'])
        ax.text(0.05, 0.95, f'Correlation: {corr:+.3f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_reward_components_combined(df):
    """Plot all reward components together for comparison"""
    reward_cols = ['rewards / match_format_exactly', 'rewards / match_format_approximately', 
                   'rewards / check_answer', 'rewards / check_numbers']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Reward Components - Combined View', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot all components together
    for col, color in zip(reward_cols, colors):
        label = col.replace('rewards / ', '').replace('_', ' ').title()
        ax1.plot(df['Step'], df[col], color=color, linewidth=1.5, alpha=0.8, label=label)
    
    # Add total reward for comparison
    ax1.plot(df['Step'], df['reward'], 'black', linewidth=2, linestyle='--', 
            label='Total Reward', alpha=0.9)
    
    ax1.set_title('All Reward Components Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot correlation heatmap
    reward_data = df[reward_cols + ['reward']].copy()
    reward_data.columns = [col.replace('rewards / ', '').replace('_', ' ').title() if 'rewards' in col else col for col in reward_data.columns]
    
    corr_matrix = reward_data.corr()
    
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Correlation Coefficient')
    
    # Set labels
    labels = list(corr_matrix.columns)
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_yticklabels(labels)
    
    # Add correlation values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    ax2.set_title('Reward Component Correlations')
    
    plt.tight_layout()
    return fig

def plot_training_context(df):
    """Plot training loss and other context metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training Context Metrics', fontsize=16, fontweight='bold')
    
    # Training Loss
    axes[0, 0].plot(df['Step'], df['Training Loss'], 'purple', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[0, 1].plot(df['Step'], df['kl'], 'brown', linewidth=2)
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('KL')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Completion Length
    axes[1, 0].plot(df['Step'], df['completion_length'], 'teal', linewidth=2)
    axes[1, 0].set_title('Completion Length')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relationship: Reward vs Training Loss
    axes[1, 1].scatter(df['Training Loss'], df['reward'], alpha=0.6, c=df['Step'], cmap='viridis')
    axes[1, 1].set_xlabel('Training Loss')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward vs Training Loss (colored by Step)')
    
    # Add colorbar for step progression
    scatter = axes[1, 1].collections[0]
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Training Step')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_summary_stats(df):
    """Generate comprehensive summary statistics"""
    print("\n" + "="*60)
    print("COMPREHENSIVE REWARD ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    print(f"Dataset: 100 training steps")
    print(f"Reward range: {df['reward'].min():.3f} to {df['reward'].max():.3f}")
    print(f"Reward mean: {df['reward'].mean():.3f} ± {df['reward'].std():.3f}")
    
    # Count positive/negative rewards
    positive_rewards = (df['reward'] > 0).sum()
    negative_rewards = (df['reward'] < 0).sum()
    zero_rewards = (df['reward'] == 0).sum()
    
    print(f"Positive rewards: {positive_rewards}/100 steps ({positive_rewards}%)")
    print(f"Negative rewards: {negative_rewards}/100 steps ({negative_rewards}%)")
    print(f"Zero rewards: {zero_rewards}/100 steps ({zero_rewards}%)")
    
    # Best performing steps
    best_steps = df.nlargest(5, 'reward')[['Step', 'reward']]
    print(f"\nTop 5 performing steps:")
    for _, row in best_steps.iterrows():
        print(f"  Step {row['Step']:3.0f}: {row['reward']:+7.3f}")
    
    # Trend analysis
    correlation = df['reward'].corr(df['Step'])
    print(f"\nTrend Analysis:")
    print(f"  Correlation with step: {correlation:+.3f}")
    
    if correlation > 0.2:
        trend = "Strong positive trend - rewards are increasing! ✓"
    elif correlation > 0.1:
        trend = "Moderate positive trend - some improvement ↗"
    elif correlation > -0.1:
        trend = "No clear trend - rewards are fluctuating ~"
    elif correlation > -0.2:
        trend = "Moderate negative trend - some degradation ↘"
    else:
        trend = "Strong negative trend - rewards are decreasing ✗"
    
    print(f"  {trend}")
    
    return correlation

def main():
    """Main function to run the complete analysis"""
    print("Loading results2.log data...")
    df = load_data()
    
    print(f"Loaded {len(df)} training steps")
    print(f"Columns: {list(df.columns)}")
    
    # Perform trend analysis
    df = analyze_reward_trends(df)
    
    # Create output directory
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Generate all plots
    fig1 = plot_reward_overview(df)
    fig1.savefig(output_dir / "reward_overview.png", dpi=300, bbox_inches='tight')
    print("✓ Reward overview saved")
    
    fig2 = plot_reward_components(df)
    fig2.savefig(output_dir / "reward_components_individual.png", dpi=300, bbox_inches='tight')
    print("✓ Individual reward components saved")
    
    fig3 = plot_reward_components_combined(df)
    fig3.savefig(output_dir / "reward_components_combined.png", dpi=300, bbox_inches='tight')
    print("✓ Combined reward components saved")
    
    fig4 = plot_training_context(df)
    fig4.savefig(output_dir / "training_context.png", dpi=300, bbox_inches='tight')
    print("✓ Training context metrics saved")
    
    # Generate final summary
    correlation = generate_summary_stats(df)
    
    # Answer the main question
    print("\n" + "="*60)
    print("ANSWER TO YOUR QUESTION:")
    print("="*60)
    if correlation > 0.1:
        print("🎉 YES! The rewards DO increase per iteration.")
        print(f"   The correlation coefficient of {correlation:+.3f} indicates")
        print("   a positive trend in reward progression.")
    else:
        print("❌ NO. The rewards do NOT clearly increase per iteration.")
        print(f"   The correlation coefficient of {correlation:+.3f} indicates")
        print("   no strong upward trend.")
    
    plt.show()

if __name__ == "__main__":
    main()
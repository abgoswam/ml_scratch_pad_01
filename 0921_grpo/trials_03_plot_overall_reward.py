import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats

def load_data():
    """Load and parse the results2.log file"""
    file_path = Path(__file__).parent / "results2.log"
    df = pd.read_csv(file_path, sep='\t')
    return df

def plot_rewards_with_trendlines(df):
    """Plot all reward columns with trendlines"""
    # Define reward columns
    reward_columns = [
        'reward',
        'rewards / match_format_exactly', 
        'rewards / match_format_approximately', 
        'rewards / check_answer', 
        'rewards / check_numbers'
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Reward Progression with Trendlines - Results2.log', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    colors = ['black', 'blue', 'green', 'red', 'orange']
    
    for i, (col, color) in enumerate(zip(reward_columns, colors)):
        ax = axes_flat[i]
        
        # Plot actual reward values
        ax.plot(df['Step'], df[col], 'o-', color=color, alpha=0.6, 
               markersize=3, linewidth=1, label='Actual Values')
        
        # Calculate and plot linear trendline
        x = df['Step'].values.reshape(-1, 1)
        y = df[col].values
        
        # Linear regression
        reg = LinearRegression().fit(x, y)
        trend_line = reg.predict(x)
        
        # Plot trendline
        ax.plot(df['Step'], trend_line, '--', color=color, linewidth=2, 
               label=f'Linear Trend (slope={reg.coef_[0]:.4f})')
        
        # Calculate correlation and p-value
        correlation, p_value = stats.pearsonr(df['Step'], df[col])
        
        # Add rolling average (smoothed trend)
        rolling_mean = df[col].rolling(window=10, center=True).mean()
        ax.plot(df['Step'], rolling_mean, '-', color=color, linewidth=2, 
               alpha=0.8, label='10-step Rolling Mean')
        
        # Formatting
        title = col.replace('rewards / ', '').replace('_', ' ').title()
        if col == 'reward':
            title = 'Total Reward'
        
        ax.set_title(f'{title}')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward Value')
        
        # Set y-axis ticks at every 0.5 interval
        y_min = df[col].min()
        y_max = df[col].max()
        y_range = y_max - y_min
        # Extend range slightly for better visualization
        y_start = np.floor(y_min * 2) / 2 - 0.5
        y_end = np.ceil(y_max * 2) / 2 + 0.5
        ax.set_yticks(np.arange(y_start, y_end + 0.1, 0.5))
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add statistics text box
        stats_text = f'Correlation: {correlation:+.3f}\nP-value: {p_value:.3f}\nSlope: {reg.coef_[0]:+.4f}'
        if p_value < 0.05:
            significance = 'Significant'
        else:
            significance = 'Not Significant'
        
        ax.text(0.02, 0.98, f'{stats_text}\n{significance}', 
               transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=8)
    
    # Remove the empty subplot
    axes_flat[5].remove()
    
    plt.tight_layout()
    return fig

def plot_combined_rewards(df):
    """Plot all reward columns together for comparison"""
    reward_columns = [
        'reward',
        'rewards / match_format_exactly', 
        'rewards / match_format_approximately', 
        'rewards / check_answer', 
        'rewards / check_numbers'
    ]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = ['black', 'blue', 'green', 'red', 'orange']
    
    for col, color in zip(reward_columns, colors):
        # Plot actual values
        label = col.replace('rewards / ', '').replace('_', ' ').title()
        if col == 'reward':
            label = 'Total Reward'
            
        ax.plot(df['Step'], df[col], 'o-', color=color, alpha=0.7, 
               markersize=2, linewidth=1, label=label)
        
        # Add trendline
        x = df['Step'].values.reshape(-1, 1)
        y = df[col].values
        reg = LinearRegression().fit(x, y)
        trend_line = reg.predict(x)
        
        ax.plot(df['Step'], trend_line, '--', color=color, linewidth=2, alpha=0.8)
    
    ax.set_title('All Reward Components with Trendlines', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Reward Value')
    
    # Set y-axis ticks at every 0.5 interval for combined plot
    all_reward_values = pd.concat([df[col] for col in reward_columns])
    y_min = all_reward_values.min()
    y_max = all_reward_values.max()
    y_start = np.floor(y_min * 2) / 2 - 0.5
    y_end = np.ceil(y_max * 2) / 2 + 0.5
    ax.set_yticks(np.arange(y_start, y_end + 0.1, 0.5))
    
    # Place legend inside the figure for better readability
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_reward_statistics(df):
    """Plot reward statistics and trend analysis"""
    reward_columns = [
        'reward',
        'rewards / match_format_exactly', 
        'rewards / match_format_approximately', 
        'rewards / check_answer', 
        'rewards / check_numbers'
    ]
    
    # Calculate statistics
    stats_data = []
    for col in reward_columns:
        x = df['Step'].values.reshape(-1, 1)
        y = df[col].values
        
        reg = LinearRegression().fit(x, y)
        correlation, p_value = stats.pearsonr(df['Step'], df[col])
        
        stats_data.append({
            'Component': col.replace('rewards / ', '').replace('_', ' ').title(),
            'Slope': reg.coef_[0],
            'Correlation': correlation,
            'P-value': p_value,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Reward Component Statistics and Trends', fontsize=16, fontweight='bold')
    
    # Slope comparison
    colors = ['black', 'blue', 'green', 'red', 'orange']
    bars1 = ax1.bar(range(len(stats_df)), stats_df['Slope'], color=colors, alpha=0.7)
    ax1.set_title('Trendline Slopes')
    ax1.set_xlabel('Reward Component')
    ax1.set_ylabel('Slope (Reward change per step)')
    ax1.set_xticks(range(len(stats_df)))
    ax1.set_xticklabels(stats_df['Component'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, slope in zip(bars1, stats_df['Slope']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.001),
                f'{slope:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # Correlation comparison
    bars2 = ax2.bar(range(len(stats_df)), stats_df['Correlation'], color=colors, alpha=0.7)
    ax2.set_title('Correlation with Training Step')
    ax2.set_xlabel('Reward Component')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_xticks(range(len(stats_df)))
    ax2.set_xticklabels(stats_df['Component'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(-1, 1)
    
    # Add value labels on bars
    for bar, corr in zip(bars2, stats_df['Correlation']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height > 0 else -0.02),
                f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # Mean and range
    x_pos = range(len(stats_df))
    ax3.errorbar(x_pos, stats_df['Mean'], yerr=stats_df['Std'], 
                fmt='o', color='blue', capsize=5, label='Mean ± Std')
    ax3.scatter(x_pos, stats_df['Min'], color='red', marker='v', s=50, label='Min')
    ax3.scatter(x_pos, stats_df['Max'], color='green', marker='^', s=50, label='Max')
    ax3.set_title('Reward Statistics Summary')
    ax3.set_xlabel('Reward Component')
    ax3.set_ylabel('Reward Value')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_df['Component'], rotation=45, ha='right')
    
    # Set y-axis ticks at every 0.5 interval for statistics plot
    y_min = min(stats_df['Min'].min(), (stats_df['Mean'] - stats_df['Std']).min())
    y_max = max(stats_df['Max'].max(), (stats_df['Mean'] + stats_df['Std']).max())
    y_start = np.floor(y_min * 2) / 2 - 0.5
    y_end = np.ceil(y_max * 2) / 2 + 0.5
    ax3.set_yticks(np.arange(y_start, y_end + 0.1, 0.5))
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # P-value significance
    bars4 = ax4.bar(range(len(stats_df)), -np.log10(stats_df['P-value']), color=colors, alpha=0.7)
    ax4.set_title('Statistical Significance (-log10 P-value)')
    ax4.set_xlabel('Reward Component')
    ax4.set_ylabel('-log10(P-value)')
    ax4.set_xticks(range(len(stats_df)))
    ax4.set_xticklabels(stats_df['Component'], rotation=45, ha='right')
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='Significance threshold (p=0.05)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, stats_df

def print_reward_analysis(df):
    """Print detailed analysis of reward trends"""
    reward_columns = [
        'reward',
        'rewards / match_format_exactly', 
        'rewards / match_format_approximately', 
        'rewards / check_answer', 
        'rewards / check_numbers'
    ]
    
    print("="*80)
    print("REWARD TREND ANALYSIS - RESULTS2.LOG")
    print("="*80)
    
    for col in reward_columns:
        x = df['Step'].values.reshape(-1, 1)
        y = df[col].values
        
        reg = LinearRegression().fit(x, y)
        correlation, p_value = stats.pearsonr(df['Step'], df[col])
        
        # Trend interpretation
        if correlation > 0.2 and p_value < 0.05:
            trend = "STRONG POSITIVE ↗↗"
        elif correlation > 0.1 and p_value < 0.05:
            trend = "MODERATE POSITIVE ↗"
        elif correlation > 0.05:
            trend = "WEAK POSITIVE ↗"
        elif correlation < -0.2 and p_value < 0.05:
            trend = "STRONG NEGATIVE ↘↘"
        elif correlation < -0.1 and p_value < 0.05:
            trend = "MODERATE NEGATIVE ↘"
        elif correlation < -0.05:
            trend = "WEAK NEGATIVE ↘"
        else:
            trend = "NEUTRAL ~"
        
        component_name = col.replace('rewards / ', '').replace('_', ' ').title()
        if col == 'reward':
            component_name = 'TOTAL REWARD'
        
        print(f"\n{component_name:.<30}")
        print(f"  Slope: {reg.coef_[0]:+.6f} per step")
        print(f"  Correlation: {correlation:+.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Trend: {trend}")
        print(f"  Range: {df[col].min():.3f} to {df[col].max():.3f}")
        print(f"  Mean ± Std: {df[col].mean():.3f} ± {df[col].std():.3f}")

def main():
    """Main function to run the reward analysis"""
    print("Loading results2.log data...")
    df = load_data()
    
    print(f"✓ Loaded {len(df)} training steps")
    
    # Print detailed analysis
    print_reward_analysis(df)
    
    # Create output directory
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating reward plots with trendlines...")
    
    # Generate plots
    fig1 = plot_rewards_with_trendlines(df)
    fig1.savefig(output_dir / "rewards_with_trendlines.png", dpi=300, bbox_inches='tight')
    print("✓ Individual reward plots with trendlines saved")
    
    fig2 = plot_combined_rewards(df)
    fig2.savefig(output_dir / "rewards_combined_trendlines.png", dpi=300, bbox_inches='tight')
    print("✓ Combined reward plot saved")
    
    fig3, stats_df = plot_reward_statistics(df)
    fig3.savefig(output_dir / "reward_statistics.png", dpi=300, bbox_inches='tight')
    print("✓ Reward statistics plot saved")
    
    # Save statistics to CSV
    stats_df.to_csv(output_dir / "reward_statistics.csv", index=False)
    print("✓ Statistics saved to CSV")
    
    print("\n" + "="*60)
    print("SUMMARY: DO REWARDS INCREASE PER ITERATION?")
    print("="*60)
    
    total_reward_corr = df['reward'].corr(df['Step'])
    if total_reward_corr > 0.1:
        print("🎉 YES! Overall rewards show an increasing trend")
        print(f"   Total reward correlation: {total_reward_corr:+.3f}")
    else:
        print("❌ NO. Overall rewards do not show a clear increasing trend")
        print(f"   Total reward correlation: {total_reward_corr:+.3f}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
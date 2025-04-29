import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_comparison_data():
    """
    Load the difficulty comparison data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_path = os.path.join(current_dir, 'results', '06_difficulty_comparison.csv')
    
    print(f"Loading comparison data from {comparison_path}")
    df = pd.read_csv(comparison_path)
    
    print(f"Loaded comparison data for {len(df)} questions")
    return df

def create_detailed_plot(df):
    """
    Create a detailed comparison plot of original vs. estimated difficulties.
    """
    # Calculate correlation and error metrics
    pearson_r, pearson_p = pearsonr(df['difficulty'], df['irt_difficulty'])
    spearman_r, spearman_p = spearmanr(df['difficulty'], df['irt_difficulty'])
    mae = np.mean(np.abs(df['difficulty'] - df['irt_difficulty']))
    rmse = np.sqrt(np.mean((df['difficulty'] - df['irt_difficulty'])**2))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color reflecting discrimination
    scatter = ax.scatter(
        df['irt_difficulty'], 
        df['difficulty'], 
        c=df['discrimination'], 
        cmap='viridis', 
        alpha=0.7,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Discrimination Parameter')
    
    # Add identity line
    min_val = min(df['irt_difficulty'].min(), df['difficulty'].min()) - 0.5
    max_val = max(df['irt_difficulty'].max(), df['difficulty'].max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Identity Line')
    
    # Add best fit line
    m, b = np.polyfit(df['irt_difficulty'], df['difficulty'], 1)
    ax.plot(
        [min_val, max_val], 
        [m * min_val + b, m * max_val + b], 
        'b-', 
        alpha=0.8,
        label=f'Best Fit Line (y = {m:.3f}x + {b:.3f})'
    )
    
    # Add correlation and error info to plot
    text_info = (
        f"Number of questions: {len(df)}\n"
        f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})\n"
        f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}"
    )
    
    # Place text box in upper left corner
    ax.text(
        0.05, 0.95, text_info, 
        transform=ax.transAxes, 
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor='gray')
    )
    
    # Label largest discrepancies
    top_discrepancies = df.nlargest(5, 'abs_diff')
    for _, row in top_discrepancies.iterrows():
        ax.annotate(
            f"Q{int(row['question_id'])}",
            xy=(row['irt_difficulty'], row['difficulty']),
            xytext=(10, 0),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
        )
    
    # Set labels and title
    ax.set_xlabel('Original IRT Difficulty')
    ax.set_ylabel('Neural Network Estimated Difficulty')
    ax.set_title('Comparison of Original vs. Neural Network Estimated Difficulty Parameters')
    
    # Set axis limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Save plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(current_dir, 'figures')
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'difficulty_comparison_detailed.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved detailed comparison plot to {plot_path}")
    
    # Show distribution of differences
    plt.figure(figsize=(12, 6))
    sns.histplot(df['difficulty'] - df['irt_difficulty'], bins=30, kde=True)
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Difference (Estimated - Original)')
    plt.ylabel('Count')
    plt.title('Distribution of Differences Between Estimated and Original Difficulties')
    plt.grid(True, alpha=0.3)
    
    # Save distribution plot
    dist_path = os.path.join(plot_dir, 'difficulty_difference_distribution.png')
    plt.tight_layout()
    plt.savefig(dist_path, dpi=300)
    print(f"Saved difference distribution plot to {dist_path}")
    
    # Create a secondary plot showing the relationship between difficulty and discrimination
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with marginal histograms
    g = sns.jointplot(
        data=df, 
        x="difficulty", 
        y="discrimination",
        kind="scatter",
        height=8,
        ratio=3,
        marginal_kws=dict(bins=30),
        joint_kws=dict(alpha=0.7, s=50)
    )
    
    # Add title
    g.fig.suptitle('Relationship Between Difficulty and Discrimination Parameters', y=1.02, fontsize=16)
    
    # Add grid
    g.ax_joint.grid(True, alpha=0.3)
    
    # Save plot
    rel_path = os.path.join(plot_dir, 'difficulty_discrimination_relationship.png')
    plt.tight_layout()
    plt.savefig(rel_path, dpi=300)
    print(f"Saved difficulty-discrimination relationship plot to {rel_path}")
    
    return fig

def main():
    # Load comparison data
    df = load_comparison_data()
    
    # Create detailed plot
    fig = create_detailed_plot(df)
    
    print("Completed generating detailed difficulty comparison plots.")
    
if __name__ == "__main__":
    main() 
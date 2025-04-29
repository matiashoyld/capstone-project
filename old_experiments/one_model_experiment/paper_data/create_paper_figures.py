import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Set plot style for publication quality - more sober, modern slate colors
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300

# Define modern slate color palette (shadcn-inspired)
SLATE_COLORS = {
    'slate-50': '#f8fafc',
    'slate-100': '#f1f5f9',
    'slate-200': '#e2e8f0',
    'slate-300': '#cbd5e1',
    'slate-400': '#94a3b8',
    'slate-500': '#64748b',
    'slate-600': '#475569',
    'slate-700': '#334155',
    'slate-800': '#1e293b',
    'slate-900': '#0f172a',
    'slate-950': '#020617'
}

# Create directory for paper figures
os.makedirs('paper_figures', exist_ok=True)

# Create directory for new minimalist figures
os.makedirs('paper_figures/minimalist', exist_ok=True)

def load_data():
    """Load all necessary data for generating paper figures."""
    data = {}
    
    # Load difficulty comparison data
    data['top100_difficulty'] = pd.read_csv('top100_difficulty_comparison.csv')
    data['all_users_difficulty'] = pd.read_csv('all_users_difficulty_comparison.csv')
    
    # Load performance metrics
    data['performance_metrics'] = pd.read_csv('performance_metrics_comparison.csv')
    data['top100_metrics'] = pd.read_csv('top100_irt_metrics.csv')
    data['all_users_metrics'] = pd.read_csv('all_users_irt_metrics.csv')
    
    # Load prediction matrices
    data['top100_proba'] = pd.read_csv('top100_probability_matrix.csv', index_col=0)
    data['all_users_proba'] = pd.read_csv('all_users_probability_matrix.csv', index_col=0)
    
    # Load user abilities
    data['top100_abilities'] = pd.read_csv('top100_user_abilities.csv', index_col=0)
    data['all_users_abilities'] = pd.read_csv('all_users_abilities.csv', index_col=0)
    
    # Load question data
    data['holdout_questions'] = pd.read_csv('holdout_questions.csv')
    
    # Load summary statistics
    data['prediction_stats'] = pd.read_csv('prediction_variation_stats.csv')
    
    return data

# New functions for minimalist figures (all users data only)

def figure_difficulty_comparison(data):
    """Create a scatter plot of actual vs estimated difficulty (without discrimination color)."""
    all_users = data['all_users_difficulty']
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with single color
    plt.scatter(
        all_users['original_difficulty'],
        all_users['estimated_difficulty'],
        alpha=0.7,
        color=SLATE_COLORS['slate-500'],
        edgecolor=SLATE_COLORS['slate-700'],
        linewidth=0.3,
        s=60
    )
    
    # Add identity line
    min_val = min(all_users['original_difficulty'].min(), all_users['estimated_difficulty'].min())
    max_val = max(all_users['original_difficulty'].max(), all_users['estimated_difficulty'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color=SLATE_COLORS['slate-400'], alpha=0.8, linewidth=1.5)
    
    # Best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        all_users['original_difficulty'],
        all_users['estimated_difficulty']
    )
    plt.plot(
        [min_val, max_val],
        [slope * min_val + intercept, slope * max_val + intercept],
        '-', color=SLATE_COLORS['slate-800'],
        alpha=0.8,
        linewidth=1.5
    )
    
    # Add correlation info
    pearson = data['all_users_metrics']['pearson_corr'].iloc[0]
    spearman = data['all_users_metrics']['spearman_corr'].iloc[0]
    plt.annotate(
        f"r = {pearson:.3f}\nρ = {spearman:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=SLATE_COLORS['slate-300'], 
                  facecolor=SLATE_COLORS['slate-50'], alpha=0.7)
    )
    
    plt.title('Actual vs. Estimated Difficulty')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Estimated Difficulty')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig1_difficulty_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig1_difficulty_comparison.pdf', bbox_inches='tight')
    plt.close()

def figure_difficulty_distribution(data):
    """Create distribution plots of actual and estimated difficulty."""
    all_users = data['all_users_difficulty']
    
    plt.figure(figsize=(12, 6))
    
    # Original difficulty distribution
    sns.kdeplot(
        all_users['original_difficulty'], 
        fill=True, 
        color=SLATE_COLORS['slate-700'], 
        alpha=0.4,
        label='Actual Difficulty',
        linewidth=2
    )
    
    # Estimated difficulty distribution
    sns.kdeplot(
        all_users['estimated_difficulty'], 
        fill=True, 
        color=SLATE_COLORS['slate-500'], 
        alpha=0.4,
        label='Estimated Difficulty',
        linewidth=2
    )
    
    # Add statistics
    orig_mean = all_users['original_difficulty'].mean()
    orig_std = all_users['original_difficulty'].std()
    est_mean = all_users['estimated_difficulty'].mean()
    est_std = all_users['estimated_difficulty'].std()
    
    plt.annotate(
        f"Actual: μ = {orig_mean:.3f}, σ = {orig_std:.3f}\nEstimated: μ = {est_mean:.3f}, σ = {est_std:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=SLATE_COLORS['slate-300'], 
                  facecolor=SLATE_COLORS['slate-50'], alpha=0.7)
    )
    
    plt.title('Distribution of Actual and Estimated Difficulty Parameters')
    plt.xlabel('Difficulty')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig2_difficulty_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig2_difficulty_distribution.pdf', bbox_inches='tight')
    plt.close()

def figure_ability_distribution(data):
    """Create a distribution plot of user abilities."""
    abilities = data['all_users_abilities']
    
    plt.figure(figsize=(10, 6))
    
    # Create the histogram with KDE
    sns.histplot(
        abilities['ability'], 
        kde=True, 
        color=SLATE_COLORS['slate-600'],
        alpha=0.6,
        bins=30,
        edgecolor=SLATE_COLORS['slate-800'],
        linewidth=0.8
    )
    
    # Add descriptive statistics
    plt.annotate(
        f"Mean: {abilities['ability'].mean():.3f}\n"
        f"Std: {abilities['ability'].std():.3f}\n"
        f"Min: {abilities['ability'].min():.3f}\n"
        f"Max: {abilities['ability'].max():.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=SLATE_COLORS['slate-300'], 
                  facecolor=SLATE_COLORS['slate-50'], alpha=0.7)
    )
    
    plt.title('Distribution of User Ability Parameters')
    plt.xlabel('Ability')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig3_ability_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig3_ability_distribution.pdf', bbox_inches='tight')
    plt.close()

def figure_probability_matrix(data):
    """Create a heatmap visualization of the probability matrix."""
    # Sample size for heatmap visualization (first n users and questions)
    sample_size = 50
    
    # Get a sample of the probability matrix
    proba_sample = data['all_users_proba'].iloc[:sample_size, :sample_size]
    
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap based on slate colors
    cmap = LinearSegmentedColormap.from_list(
        'slate_cmap', 
        [SLATE_COLORS['slate-50'], SLATE_COLORS['slate-900']], 
        N=256
    )
    
    # Create the heatmap
    ax = sns.heatmap(
        proba_sample, 
        cmap=cmap, 
        vmin=0, 
        vmax=1, 
        cbar_kws={'label': 'Probability of Correct Answer'}
    )
    
    plt.title('Probability Matrix (Sample of Users and Questions)')
    plt.xlabel('Question Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig4_probability_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig4_probability_matrix.pdf', bbox_inches='tight')
    plt.close()

def figure_correctness_matrix(data):
    """Create a binary heatmap visualization of the correctness matrix."""
    # For this example, we'll convert probabilities to binary correctness
    # using a threshold of 0.5 (probability > 0.5 is considered correct)
    sample_size = 50
    
    # Get a sample of the probability matrix
    proba_sample = data['all_users_proba'].iloc[:sample_size, :sample_size]
    
    # Convert to binary correctness matrix
    correctness_sample = (proba_sample > 0.5).astype(int)
    
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'correctness_cmap', 
        [SLATE_COLORS['slate-200'], SLATE_COLORS['slate-800']], 
        N=2
    )
    
    # Create the heatmap
    ax = sns.heatmap(
        correctness_sample, 
        cmap=cmap, 
        vmin=0, 
        vmax=1, 
        cbar_kws={'label': 'Correct (1) or Incorrect (0)'},
        linewidths=0.5,
        linecolor=SLATE_COLORS['slate-100']
    )
    
    plt.title('Correctness Matrix (Sample of Users and Questions)')
    plt.xlabel('Question Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig5_correctness_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig5_correctness_matrix.pdf', bbox_inches='tight')
    plt.close()

def figure_full_probability_matrix(data):
    """Create a heatmap visualization of the complete probability matrix with all users and questions."""
    # Get the full probability matrix
    full_proba = data['all_users_proba']
    
    # Due to the large size, we'll create a downsampled visualization
    # The idea is to represent the overall pattern without trying to show individual cells
    plt.figure(figsize=(16, 12))
    
    # Create a custom colormap based on slate colors
    cmap = LinearSegmentedColormap.from_list(
        'slate_cmap', 
        [SLATE_COLORS['slate-50'], SLATE_COLORS['slate-900']], 
        N=256
    )
    
    # Use imshow for better performance with large matrices
    plt.imshow(
        full_proba, 
        cmap=cmap, 
        aspect='auto',  # Auto aspect ratio to fit the figure
        interpolation='nearest',  # No interpolation between pixels
        vmin=0, 
        vmax=1
    )
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Probability of Correct Answer')
    
    # Add stats as text annotation
    mean_prob = full_proba.values.mean()
    std_prob = full_proba.values.std()
    plt.annotate(
        f"Mean: {mean_prob:.3f}\nStd: {std_prob:.3f}\nShape: {full_proba.shape[0]} × {full_proba.shape[1]}",
        xy=(0.02, 0.97),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=SLATE_COLORS['slate-300'], 
                  facecolor=SLATE_COLORS['slate-50'], alpha=0.7)
    )
    
    plt.title('Full Probability Matrix (All Users and Questions)')
    plt.xlabel('Question Index')
    plt.ylabel('User Index')
    
    # Remove actual tick labels as they would be too crowded
    plt.xticks([])
    plt.yticks([])
    
    # Add dimension indicators at the axes
    plt.text(full_proba.shape[1]/2, -10, f"{full_proba.shape[1]} Questions", 
             ha='center', va='top', fontsize=10)
    plt.text(-10, full_proba.shape[0]/2, f"{full_proba.shape[0]} Users", 
             ha='right', va='center', fontsize=10, rotation=90)
    
    plt.tight_layout()
    
    plt.savefig('paper_figures/minimalist/fig6_full_probability_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/minimalist/fig6_full_probability_matrix.pdf', bbox_inches='tight')
    plt.close()

def generate_minimalist_figures():
    """Generate all minimalist figures using all users data."""
    print("Loading data...")
    data = load_data()
    
    print("Generating Figure 1: Actual vs Estimated Difficulty Comparison...")
    figure_difficulty_comparison(data)
    
    print("Generating Figure 2: Difficulty Distribution...")
    figure_difficulty_distribution(data)
    
    print("Generating Figure 3: User Ability Distribution...")
    figure_ability_distribution(data)
    
    print("Generating Figure 4: Probability Matrix...")
    figure_probability_matrix(data)
    
    print("Generating Figure 5: Correctness Matrix...")
    figure_correctness_matrix(data)
    
    print("Generating Figure 6: Full Probability Matrix...")
    figure_full_probability_matrix(data)
    
    print("\nAll minimalist figures saved in 'paper_figures/minimalist' directory.")
    print("PDF and PNG formats are provided for each figure.")

# Keep the original functions below

def figure1_difficulty_comparison(data):
    """Create a side-by-side comparison of difficulty estimates for top 100 users vs all users."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True, sharex=True)
    
    # Define a custom colormap
    cmap = plt.cm.viridis
    
    # Top 100 Users plot
    scatter1 = axs[0].scatter(
        data['top100_difficulty']['original_difficulty'],
        data['top100_difficulty']['estimated_difficulty'],
        c=data['top100_difficulty']['discrimination'],
        cmap=cmap,
        alpha=0.7,
        s=50,
        edgecolor='black',
        linewidth=0.3
    )
    
    # Add identity line
    min_val = min(data['top100_difficulty']['original_difficulty'].min(), 
                data['top100_difficulty']['estimated_difficulty'].min())
    max_val = max(data['top100_difficulty']['original_difficulty'].max(), 
                data['top100_difficulty']['estimated_difficulty'].max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5)
    
    # Best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data['top100_difficulty']['original_difficulty'],
        data['top100_difficulty']['estimated_difficulty']
    )
    axs[0].plot(
        [min_val, max_val],
        [slope * min_val + intercept, slope * max_val + intercept],
        'b-',
        alpha=0.8,
        linewidth=1.5
    )
    
    # Add correlation info
    pearson = data['top100_metrics']['pearson_corr'].iloc[0]
    spearman = data['top100_metrics']['spearman_corr'].iloc[0]
    axs[0].annotate(
        f"r = {pearson:.3f}\nρ = {spearman:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    axs[0].set_title('Top 100 Users')
    axs[0].set_xlabel('Original IRT Difficulty')
    axs[0].set_ylabel('Estimated Difficulty')
    
    # All Users plot
    scatter2 = axs[1].scatter(
        data['all_users_difficulty']['original_difficulty'],
        data['all_users_difficulty']['estimated_difficulty'],
        c=data['all_users_difficulty']['discrimination'],
        cmap=cmap,
        alpha=0.7,
        s=50,
        edgecolor='black',
        linewidth=0.3
    )
    
    # Add identity line
    axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5)
    
    # Best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data['all_users_difficulty']['original_difficulty'],
        data['all_users_difficulty']['estimated_difficulty']
    )
    axs[1].plot(
        [min_val, max_val],
        [slope * min_val + intercept, slope * max_val + intercept],
        'b-',
        alpha=0.8,
        linewidth=1.5
    )
    
    # Add correlation info
    pearson = data['all_users_metrics']['pearson_corr'].iloc[0]
    spearman = data['all_users_metrics']['spearman_corr'].iloc[0]
    axs[1].annotate(
        f"r = {pearson:.3f}\nρ = {spearman:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    axs[1].set_title('All Users (1,867)')
    axs[1].set_xlabel('Original IRT Difficulty')
    
    # Add a colorbar
    cbar = fig.colorbar(scatter2, ax=axs, pad=0.01)
    cbar.set_label('Discrimination Parameter')
    
    plt.suptitle('Comparison of Original vs Estimated IRT Difficulties', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('paper_figures/fig1_difficulty_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig1_difficulty_comparison.pdf', bbox_inches='tight')
    plt.close()

def figure2_metrics_comparison(data):
    """Create a comparison of key metrics between the two approaches."""
    metrics = data['performance_metrics']
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot correlation metrics
    correlation_metrics = ['Pearson Correlation', 'Spearman Correlation']
    x = np.arange(len(correlation_metrics))
    width = 0.35
    
    correlation_values = {
        'Top 100 Users': [metrics.loc[metrics['Metric'] == 'Pearson Correlation', 'Top 100 Users'].iloc[0],
                        metrics.loc[metrics['Metric'] == 'Spearman Correlation', 'Top 100 Users'].iloc[0]],
        'All Users': [metrics.loc[metrics['Metric'] == 'Pearson Correlation', 'All Users'].iloc[0],
                    metrics.loc[metrics['Metric'] == 'Spearman Correlation', 'All Users'].iloc[0]]
    }
    
    ax1.bar(x - width/2, correlation_values['Top 100 Users'], width, label='Top 100 Users', 
           color='#3498db', edgecolor='black', linewidth=1)
    ax1.bar(x + width/2, correlation_values['All Users'], width, label='All Users', 
           color='#e74c3c', edgecolor='black', linewidth=1)
    
    # Add value labels on the bars
    for i, v in enumerate(correlation_values['Top 100 Users']):
        ax1.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(correlation_values['All Users']):
        ax1.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Correlation Value')
    ax1.set_title('Correlation Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(correlation_metrics)
    ax1.set_ylim(0, 1.1)  # Set y-axis limits
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot error metrics
    error_metrics = ['MAE', 'RMSE']
    x = np.arange(len(error_metrics))
    
    error_values = {
        'Top 100 Users': [metrics.loc[metrics['Metric'] == 'MAE', 'Top 100 Users'].iloc[0],
                         metrics.loc[metrics['Metric'] == 'RMSE', 'Top 100 Users'].iloc[0]],
        'All Users': [metrics.loc[metrics['Metric'] == 'MAE', 'All Users'].iloc[0],
                     metrics.loc[metrics['Metric'] == 'RMSE', 'All Users'].iloc[0]]
    }
    
    ax2.bar(x - width/2, error_values['Top 100 Users'], width, label='Top 100 Users', 
           color='#3498db', edgecolor='black', linewidth=1)
    ax2.bar(x + width/2, error_values['All Users'], width, label='All Users', 
           color='#e74c3c', edgecolor='black', linewidth=1)
    
    # Add value labels on the bars
    for i, v in enumerate(error_values['Top 100 Users']):
        ax2.text(i - width/2, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(error_values['All Users']):
        ax2.text(i + width/2, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Error Value')
    ax2.set_title('Error Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_metrics)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend to the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              fancybox=True, shadow=True, ncol=2)
    
    plt.suptitle('Performance Metrics Comparison', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('paper_figures/fig2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig2_metrics_comparison.pdf', bbox_inches='tight')
    plt.close()

def figure3_prediction_variation(data):
    """Create visualizations showing the variation in predictions."""
    # Sample size for heatmap visualization (first n users and questions)
    sample_size = 30
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Top 100 Users Heatmap
    top100_sample = data['top100_proba'].iloc[:sample_size, :sample_size]
    sns.heatmap(top100_sample, cmap='viridis', vmin=0, vmax=1, 
               cbar_kws={'label': 'Probability'}, ax=axs[0, 0])
    axs[0, 0].set_title('Top 100 Users: Probability Matrix Sample')
    axs[0, 0].set_xlabel('Question Index')
    axs[0, 0].set_ylabel('User Index')
    
    # All Users Heatmap
    all_users_sample = data['all_users_proba'].iloc[:sample_size, :sample_size]
    sns.heatmap(all_users_sample, cmap='viridis', vmin=0, vmax=1, 
               cbar_kws={'label': 'Probability'}, ax=axs[0, 1])
    axs[0, 1].set_title('All Users: Probability Matrix Sample')
    axs[0, 1].set_xlabel('Question Index')
    axs[0, 1].set_ylabel('User Index')
    
    # Standard Deviation Comparison - As a bar chart
    stats = data['prediction_stats']
    
    metrics = ['Std_across_questions', 'Std_across_users']
    x = np.arange(len(metrics))
    width = 0.35
    
    axs[1, 0].bar(x - width/2, stats.loc[stats['Model'] == 'Top 100 Users', metrics].values[0], 
                 width, label='Top 100 Users', color='#3498db', edgecolor='black', linewidth=1)
    axs[1, 0].bar(x + width/2, stats.loc[stats['Model'] == 'All Users', metrics].values[0], 
                 width, label='All Users', color='#e74c3c', edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, model in enumerate(['Top 100 Users', 'All Users']):
        for j, metric in enumerate(metrics):
            value = stats.loc[stats['Model'] == model, metric].values[0]
            axs[1, 0].text(j + (i-0.5)*width, value + 0.01, f'{value:.3f}', 
                          ha='center', va='bottom', fontsize=10)
    
    axs[1, 0].set_title('Prediction Variation')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(['Across Questions', 'Across Users'])
    axs[1, 0].set_ylabel('Standard Deviation')
    axs[1, 0].legend()
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Probability distribution comparison
    top100_values = data['top100_proba'].values.flatten()
    all_users_values = data['all_users_proba'].values.flatten()
    
    # Sample for faster plotting if needed
    if len(top100_values) > 10000:
        np.random.seed(42)
        top100_values = np.random.choice(top100_values, 10000, replace=False)
    if len(all_users_values) > 10000:
        np.random.seed(42)
        all_users_values = np.random.choice(all_users_values, 10000, replace=False)
    
    sns.kdeplot(top100_values, fill=True, color='#3498db', alpha=0.5, 
               label='Top 100 Users', ax=axs[1, 1])
    sns.kdeplot(all_users_values, fill=True, color='#e74c3c', alpha=0.5, 
               label='All Users', ax=axs[1, 1])
    
    axs[1, 1].set_title('Distribution of Prediction Probabilities')
    axs[1, 1].set_xlabel('Probability')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1, 1].legend()
    
    plt.suptitle('Prediction Variation Analysis', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('paper_figures/fig3_prediction_variation.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig3_prediction_variation.pdf', bbox_inches='tight')
    plt.close()

def figure4_ability_distribution(data):
    """Create visualizations showing the distribution of user abilities."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 100 Users Ability Distribution
    sns.histplot(data['top100_abilities']['ability'], kde=True, color='#3498db', 
                bins=20, ax=axs[0])
    axs[0].set_title('Top 100 Users: Ability Distribution')
    axs[0].set_xlabel('Ability Parameter')
    axs[0].set_ylabel('Count')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add descriptive statistics
    axs[0].annotate(
        f"Mean: {data['top100_abilities']['ability'].mean():.3f}\n"
        f"Std: {data['top100_abilities']['ability'].std():.3f}\n"
        f"Min: {data['top100_abilities']['ability'].min():.3f}\n"
        f"Max: {data['top100_abilities']['ability'].max():.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    # All Users Ability Distribution
    sns.histplot(data['all_users_abilities']['ability'], kde=True, color='#e74c3c', 
                bins=30, ax=axs[1])
    axs[1].set_title('All Users: Ability Distribution')
    axs[1].set_xlabel('Ability Parameter')
    axs[1].set_ylabel('Count')
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add descriptive statistics
    axs[1].annotate(
        f"Mean: {data['all_users_abilities']['ability'].mean():.3f}\n"
        f"Std: {data['all_users_abilities']['ability'].std():.3f}\n"
        f"Min: {data['all_users_abilities']['ability'].min():.3f}\n"
        f"Max: {data['all_users_abilities']['ability'].max():.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    plt.suptitle('User Ability Parameter Distributions', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('paper_figures/fig4_ability_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig4_ability_distribution.pdf', bbox_inches='tight')
    plt.close()

def figure5_difficulty_error_analysis(data):
    """Create visualizations analyzing the errors in difficulty estimation."""
    # Prepare data
    top100 = data['top100_difficulty'].copy()
    all_users = data['all_users_difficulty'].copy()
    
    top100['error'] = top100['estimated_difficulty'] - top100['original_difficulty']
    all_users['error'] = all_users['estimated_difficulty'] - all_users['original_difficulty']
    
    # Create the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top 100 Users: Error vs Original Difficulty
    axs[0, 0].scatter(top100['original_difficulty'], top100['error'], 
                    alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.3)
    axs[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    
    # Add best fit line
    z = np.polyfit(top100['original_difficulty'], top100['error'], 1)
    p = np.poly1d(z)
    axs[0, 0].plot(top100['original_difficulty'], p(top100['original_difficulty']), 
                 'k--', alpha=0.8, linewidth=1.5)
    
    axs[0, 0].set_title('Top 100 Users: Error vs Original Difficulty')
    axs[0, 0].set_xlabel('Original Difficulty')
    axs[0, 0].set_ylabel('Error (Estimated - Original)')
    axs[0, 0].grid(alpha=0.3)
    
    # All Users: Error vs Original Difficulty
    axs[0, 1].scatter(all_users['original_difficulty'], all_users['error'], 
                    alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.3)
    axs[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    
    # Add best fit line
    z = np.polyfit(all_users['original_difficulty'], all_users['error'], 1)
    p = np.poly1d(z)
    axs[0, 1].plot(all_users['original_difficulty'], p(all_users['original_difficulty']), 
                 'k--', alpha=0.8, linewidth=1.5)
    
    axs[0, 1].set_title('All Users: Error vs Original Difficulty')
    axs[0, 1].set_xlabel('Original Difficulty')
    axs[0, 1].set_ylabel('Error (Estimated - Original)')
    axs[0, 1].grid(alpha=0.3)
    
    # Error Distribution: Top 100 Users
    sns.histplot(top100['error'], kde=True, color='#3498db', ax=axs[1, 0], bins=20)
    axs[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axs[1, 0].axvline(x=top100['error'].mean(), color='g', linestyle='-', 
                    label=f'Mean: {top100["error"].mean():.3f}')
    axs[1, 0].axvline(x=top100['error'].median(), color='b', linestyle='-.', 
                    label=f'Median: {top100["error"].median():.3f}')
    
    axs[1, 0].set_title('Top 100 Users: Error Distribution')
    axs[1, 0].set_xlabel('Error (Estimated - Original)')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Error Distribution: All Users
    sns.histplot(all_users['error'], kde=True, color='#e74c3c', ax=axs[1, 1], bins=20)
    axs[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axs[1, 1].axvline(x=all_users['error'].mean(), color='g', linestyle='-', 
                    label=f'Mean: {all_users["error"].mean():.3f}')
    axs[1, 1].axvline(x=all_users['error'].median(), color='b', linestyle='-.', 
                    label=f'Median: {all_users["error"].median():.3f}')
    
    axs[1, 1].set_title('All Users: Error Distribution')
    axs[1, 1].set_xlabel('Error (Estimated - Original)')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Difficulty Estimation Error Analysis', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('paper_figures/fig5_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig5_error_analysis.pdf', bbox_inches='tight')
    plt.close()

def figure6_discrimination_analysis(data):
    """Create visualizations analyzing the discrimination parameters."""
    # Prepare data
    top100 = data['top100_difficulty'].copy()
    all_users = data['all_users_difficulty'].copy()
    
    # Calculate discrepancy (as absolute difference between estimated and original difficulty)
    top100['discrepancy'] = abs(top100['estimated_difficulty'] - top100['original_difficulty'])
    all_users['discrepancy'] = abs(all_users['estimated_difficulty'] - all_users['original_difficulty'])
    
    # Create the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top 100 Users: Discrimination vs Original Difficulty
    scatter1 = axs[0, 0].scatter(top100['original_difficulty'], top100['discrimination'], 
                             alpha=0.7, c=top100['discrepancy'], cmap='viridis', 
                             edgecolor='black', linewidth=0.3)
    
    # Add best fit line
    z = np.polyfit(top100['original_difficulty'], top100['discrimination'], 1)
    p = np.poly1d(z)
    axs[0, 0].plot(top100['original_difficulty'], p(top100['original_difficulty']), 
                 'r--', alpha=0.8, linewidth=1.5)
    
    # Add correlation information
    r = np.corrcoef(top100['original_difficulty'], top100['discrimination'])[0, 1]
    axs[0, 0].annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top', 
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
    
    axs[0, 0].set_title('Top 100 Users: Discrimination vs Difficulty')
    axs[0, 0].set_xlabel('Original Difficulty')
    axs[0, 0].set_ylabel('Discrimination Parameter')
    axs[0, 0].grid(alpha=0.3)
    plt.colorbar(scatter1, ax=axs[0, 0], label='Discrepancy')
    
    # All Users: Discrimination vs Original Difficulty
    scatter2 = axs[0, 1].scatter(all_users['original_difficulty'], all_users['discrimination'], 
                             alpha=0.7, c=all_users['discrepancy'], cmap='viridis', 
                             edgecolor='black', linewidth=0.3)
    
    # Add best fit line
    z = np.polyfit(all_users['original_difficulty'], all_users['discrimination'], 1)
    p = np.poly1d(z)
    axs[0, 1].plot(all_users['original_difficulty'], p(all_users['original_difficulty']), 
                 'r--', alpha=0.8, linewidth=1.5)
    
    # Add correlation information
    r = np.corrcoef(all_users['original_difficulty'], all_users['discrimination'])[0, 1]
    axs[0, 1].annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top', 
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
    
    axs[0, 1].set_title('All Users: Discrimination vs Difficulty')
    axs[0, 1].set_xlabel('Original Difficulty')
    axs[0, 1].set_ylabel('Discrimination Parameter')
    axs[0, 1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axs[0, 1], label='Discrepancy')
    
    # Discrimination Distribution: Top 100 Users
    sns.histplot(top100['discrimination'], kde=True, color='#3498db', ax=axs[1, 0], bins=20)
    axs[1, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.8, 
                    label='Boundary (a=1.0)')
    axs[1, 0].axvline(x=top100['discrimination'].mean(), color='g', linestyle='-', 
                    label=f'Mean: {top100["discrimination"].mean():.3f}')
    
    axs[1, 0].set_title('Top 100 Users: Discrimination Distribution')
    axs[1, 0].set_xlabel('Discrimination Parameter')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Discrimination Distribution: All Users
    sns.histplot(all_users['discrimination'], kde=True, color='#e74c3c', ax=axs[1, 1], bins=20)
    axs[1, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.8, 
                    label='Boundary (a=1.0)')
    axs[1, 1].axvline(x=all_users['discrimination'].mean(), color='g', linestyle='-', 
                    label=f'Mean: {all_users["discrimination"].mean():.3f}')
    
    axs[1, 1].set_title('All Users: Discrimination Distribution')
    axs[1, 1].set_xlabel('Discrimination Parameter')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Discrimination Parameter Analysis', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('paper_figures/fig6_discrimination_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/fig6_discrimination_analysis.pdf', bbox_inches='tight')
    plt.close()

def generate_all_figures():
    """Generate all figures for the paper."""
    print("Loading data...")
    data = load_data()
    
    print("Generating Figure 1: Difficulty Comparison...")
    figure1_difficulty_comparison(data)
    
    print("Generating Figure 2: Metrics Comparison...")
    figure2_metrics_comparison(data)
    
    print("Generating Figure 3: Prediction Variation...")
    figure3_prediction_variation(data)
    
    print("Generating Figure 4: Ability Distribution...")
    figure4_ability_distribution(data)
    
    print("Generating Figure 5: Error Analysis...")
    figure5_difficulty_error_analysis(data)
    
    print("Generating Figure 6: Discrimination Analysis...")
    figure6_discrimination_analysis(data)
    
    print("\nAll figures saved in the 'paper_figures' directory.")
    print("PDF and PNG formats are provided for each figure.")

# Update the main execution section
if __name__ == "__main__":
    # Generate the minimalist figures only
    generate_minimalist_figures()
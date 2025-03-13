import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import os
from scipy import stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colors import TAILWIND_COLORS
# Set style parameters for all plots
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    """
    Load the processed data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df

def stratified_question_split(df, test_size=0.2, random_state=42):
    """
    Split questions into train and test sets in a stratified manner based on 
    correct answer ratios to ensure balanced distributions.
    
    Returns question_ids for train and test sets.
    """
    # Get unique questions and their metrics
    question_df = df.groupby('question_id').agg({
        'is_correct': 'mean'  # Average correctness rate per question
    }).reset_index()
    
    # Bin correctness rate into 5 bins
    question_df['correctness_bin'] = pd.qcut(question_df['is_correct'], 5, labels=False)
    
    # Split the questions using stratification
    train_questions, test_questions = train_test_split(
        question_df,
        test_size=test_size,
        random_state=random_state,
        stratify=question_df['correctness_bin']
    )
    
    return train_questions['question_id'].tolist(), test_questions['question_id'].tolist()

def prepare_data_for_visualization(df):
    """
    Prepare data splits for visualization.
    """
    # Get train/test split
    train_question_ids, test_question_ids = stratified_question_split(df)
    
    # Split data based on question IDs
    train_df = df[df['question_id'].isin(train_question_ids)].copy()
    test_df = df[df['question_id'].isin(test_question_ids)].copy()
    
    # Further split train into train and validation (80/20 of train)
    train_question_ids, val_question_ids = train_test_split(
        train_question_ids,
        test_size=0.2,
        random_state=42
    )
    
    # Update train_df and create val_df
    val_df = train_df[train_df['question_id'].isin(val_question_ids)].copy()
    train_df = train_df[train_df['question_id'].isin(train_question_ids)].copy()
    
    # Add dataset labels
    train_df['split'] = 'Train'
    val_df['split'] = 'Validation'
    test_df['split'] = 'Test'
    
    # Combine all for easy plotting
    all_df = pd.concat([train_df, val_df, test_df])
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, val_df, test_df, all_df

def plot_difficulty_distribution(train_df, val_df, test_df):
    """
    Plot KDE of difficulty distribution across train, validation, and test sets.
    Only considers unique questions.
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique question data
    train_questions = train_df.drop_duplicates('question_id')
    val_questions = val_df.drop_duplicates('question_id')
    test_questions = test_df.drop_duplicates('question_id')
    
    print(f"Unique questions - Train: {len(train_questions)}, Validation: {len(val_questions)}, Test: {len(test_questions)}")
    
    # Plot filled KDE for each split with lower alpha
    sns.kdeplot(train_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-900'], 
               label='Train', alpha=0.2, fill=True)
    sns.kdeplot(val_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-700'], 
               label='Validation', alpha=0.2, fill=True)
    sns.kdeplot(test_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-500'], 
               label='Test', alpha=0.2, fill=True)
    
    # Plot line KDE with alpha=1 on top (border)
    sns.kdeplot(train_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-900'], 
               alpha=1, fill=False, label='_nolegend_')
    sns.kdeplot(val_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-700'], 
               alpha=1, fill=False, label='_nolegend_')
    sns.kdeplot(test_questions['irt_difficulty'], color=TAILWIND_COLORS['slate-500'], 
               alpha=1, fill=False, label='_nolegend_')
    
    # Add statistics
    train_mean = train_questions['irt_difficulty'].mean()
    train_std = train_questions['irt_difficulty'].std()
    val_mean = val_questions['irt_difficulty'].mean()
    val_std = val_questions['irt_difficulty'].std()
    test_mean = test_questions['irt_difficulty'].mean()
    test_std = test_questions['irt_difficulty'].std()
    
    plt.annotate(
        f"Train: μ = {train_mean:.3f}, σ = {train_std:.3f}\nValidation: μ = {val_mean:.3f}, σ = {val_std:.3f}\nTest: μ = {test_mean:.3f}, σ = {test_std:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=TAILWIND_COLORS['slate-300'], 
                 facecolor='white', alpha=0.7)
    )
    
    plt.xlabel('IRT Difficulty')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'difficulty_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_is_correct_distribution(train_df, val_df, test_df, all_df):
    """
    Plot stacked bar chart of is_correct distribution across train, validation, test sets, and overall.
    Only considers unique questions.
    """
    # Get unique question data with average correctness
    train_questions = train_df.groupby('question_id')['is_correct'].mean().reset_index()
    val_questions = val_df.groupby('question_id')['is_correct'].mean().reset_index()
    test_questions = test_df.groupby('question_id')['is_correct'].mean().reset_index()
    all_questions = all_df.groupby('question_id')['is_correct'].mean().reset_index()
    
    # Calculate percentage of correct answers for each split
    correct_percentages = []
    for df, label in [(train_questions, 'Train'), (val_questions, 'Validation'), 
                      (test_questions, 'Test'), (all_questions, 'All')]:
        correct = df['is_correct'].mean() * 100
        incorrect = 100 - correct
        correct_percentages.append({'Split': label, 'Correct': correct, 'Incorrect': incorrect})
    
    result_df = pd.DataFrame(correct_percentages)
    
    # Create stacked bar chart
    plt.figure(figsize=(10, 6))
    
    # Create positions for the bars
    positions = np.arange(len(result_df['Split']))
    
    # Plot stacked bars
    incorrect_bars = plt.bar(positions, result_df['Incorrect'], color=TAILWIND_COLORS['slate-500'], 
            label='Incorrect', edgecolor='white')
    correct_bars = plt.bar(positions, result_df['Correct'], bottom=result_df['Incorrect'], 
            color=TAILWIND_COLORS['slate-400'], label='Correct', edgecolor='white')
    
    # Add percentage labels
    for i, (incorrect, correct) in enumerate(zip(incorrect_bars, correct_bars)):
        # Format to one decimal place
        incorrect_height = round(result_df['Incorrect'].iloc[i], 1)
        correct_height = round(result_df['Correct'].iloc[i], 1)
        
        # Add the percentage labels
        plt.text(incorrect.get_x() + incorrect.get_width()/2, incorrect_height/2,
                 f'{incorrect_height:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        
        plt.text(correct.get_x() + correct.get_width()/2, incorrect_height + correct_height/2,
                 f'{correct_height:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    # Customize plot
    plt.xticks(positions, result_df['Split'])
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'is_correct_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_level_distribution(all_df):
    """
    Plot distribution of user levels from the merged_features_filtered.csv file.
    Groups by user_id and takes the average level for each user.
    Uses smaller bins and adds a KDE curve overlay.
    """
    plt.figure(figsize=(12, 7))
    
    # Ensure user_level is float
    all_df['user_level'] = all_df['user_level'].astype(float)
    
    # Group by user_id and calculate average user_level
    unique_users = all_df.groupby('user_id')['user_level'].mean().reset_index()
    
    # Filter out NaN and infinite values
    unique_users = unique_users[np.isfinite(unique_users['user_level'])]
    
    print(f"Grouped data by user_id: {len(unique_users)} unique users")
    print(f"User level range: Min={unique_users['user_level'].min()}, Max={unique_users['user_level'].max()}")
    
    # Determine bin range based on actual data range with some padding
    min_level = float(np.floor(unique_users['user_level'].min() * 4) / 4)  # Round down to nearest 0.25
    max_level = float(np.ceil(unique_users['user_level'].max() * 4) / 4)   # Round up to nearest 0.25
    
    # Create smaller bins (width of 0.2) for finer granularity
    bins = np.arange(min_level, max_level + 0.21, 0.2)  # +0.21 to ensure the last bin is included
    
    # Set up figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Create histogram with smaller bins
    n, bins, patches = ax1.hist(unique_users['user_level'], bins=bins, 
                              color=TAILWIND_COLORS['slate-400'], 
                              edgecolor='white', label='Histogram')
    
    # Create a second y-axis for the KDE
    ax2 = ax1.twinx()
    
    # Calculate KDE manually for more control
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(unique_users['user_level'].values)
        x_range = np.linspace(min_level, max_level, 1000)
        kde_values = kde(x_range)
        
        # Plot KDE curve on second axis
        ax2.plot(x_range, kde_values, color=TAILWIND_COLORS['slate-700'], linewidth=3, label='Density')
        
        # Adjust tick colors for KDE axis
        ax2.tick_params(axis='y', colors=TAILWIND_COLORS['slate-700'])
        
        # Force zero to be included in KDE axis range
        ax2.set_ylim(0, ax2.get_ylim()[1])
    except Exception as e:
        print(f"Warning: Could not create KDE curve due to error: {e}")
        # Hide the right y-axis if KDE fails
        ax2.set_visible(False)
    
    # Set labels and title
    ax1.set_xlabel('User Level', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14, color=TAILWIND_COLORS['slate-700'])
    
    # Remove y-label from right axis
    ax2.set_ylabel('Density', fontsize=14, color=TAILWIND_COLORS['slate-700'])
    
    # Adjust tick colors for histogram axis
    ax1.tick_params(axis='y', colors=TAILWIND_COLORS['slate-700'])
    
    
    # Add grid
    ax1.grid(axis='y', alpha=0.3)
    
    # Force zero to be included in histogram axis range
    ax1.set_ylim(0, ax1.get_ylim()[1])
    
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'user_level_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def load_difficulty_comparison_data():
    """
    Load the difficulty comparison data from the paper_data directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'paper_data', 'all_users_difficulty_comparison.csv')
    metrics_path = os.path.join(current_dir, 'paper_data', 'all_users_irt_metrics.csv')
    
    if not os.path.exists(data_path) or not os.path.exists(metrics_path):
        print("Warning: Difficulty comparison data not found.")
        return None, None
    
    print(f"Loading difficulty comparison data from {data_path}")
    difficulty_df = pd.read_csv(data_path)
    
    print(f"Loading IRT metrics from {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    
    return difficulty_df, metrics_df

def plot_difficulty_comparison(difficulty_df, metrics_df):
    """
    Create a scatter plot comparing actual vs estimated difficulty.
    """
    if difficulty_df is None or metrics_df is None:
        print("Skipping difficulty comparison plot due to missing data.")
        return
    
    plt.figure(figsize=(8, 8))
    
    # Create scatter plot with single color
    plt.scatter(
        difficulty_df['original_difficulty'],
        difficulty_df['estimated_difficulty'],
        alpha=0.7,
        color=TAILWIND_COLORS['slate-500'],
        edgecolor=TAILWIND_COLORS['slate-700'],
        linewidth=0.3,
        s=60
    )
    
    # Add identity line
    min_val = min(difficulty_df['original_difficulty'].min(), difficulty_df['estimated_difficulty'].min())
    max_val = max(difficulty_df['original_difficulty'].max(), difficulty_df['estimated_difficulty'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color=TAILWIND_COLORS['slate-400'], alpha=0.8, linewidth=1.5)
    
    # Best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        difficulty_df['original_difficulty'],
        difficulty_df['estimated_difficulty']
    )
    plt.plot(
        [min_val, max_val],
        [slope * min_val + intercept, slope * max_val + intercept],
        '-', color=TAILWIND_COLORS['slate-800'],
        alpha=0.8,
        linewidth=1.5
    )
    
    # Add correlation info
    pearson = metrics_df['pearson_corr'].iloc[0]
    spearman = metrics_df['spearman_corr'].iloc[0]
    plt.annotate(
        f"r = {pearson:.3f}\nρ = {spearman:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=TAILWIND_COLORS['slate-300'], 
                 facecolor='white', alpha=0.7)
    )

    plt.xlabel('Actual Difficulty')
    plt.ylabel('Estimated Difficulty')
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'difficulty_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_difficulty_distributions(difficulty_df):
    """
    Create distribution plots of actual and estimated difficulty parameters.
    """
    if difficulty_df is None:
        print("Skipping difficulty distributions plot due to missing data.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Original difficulty distribution
    sns.kdeplot(
        difficulty_df['original_difficulty'], 
        fill=True, 
        color=TAILWIND_COLORS['slate-700'], 
        alpha=0.3,
        label='Actual Difficulty',
        linewidth=2
    )
    
    # Estimated difficulty distribution
    sns.kdeplot(
        difficulty_df['estimated_difficulty'], 
        fill=True, 
        color=TAILWIND_COLORS['slate-500'], 
        alpha=0.3,
        label='Estimated Difficulty',
        linewidth=2
    )
    
    # Add statistics
    orig_mean = difficulty_df['original_difficulty'].mean()
    orig_std = difficulty_df['original_difficulty'].std()
    est_mean = difficulty_df['estimated_difficulty'].mean()
    est_std = difficulty_df['estimated_difficulty'].std()
    
    plt.annotate(
        f"Actual: μ = {orig_mean:.3f}, σ = {orig_std:.3f}\nEstimated: μ = {est_mean:.3f}, σ = {est_std:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor=TAILWIND_COLORS['slate-300'], 
                 facecolor='white', alpha=0.7)
    )
    

    plt.xlabel('Difficulty')
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'difficulty_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    df = load_data()
    
    # Prepare data for visualization
    train_df, val_df, test_df, all_df = prepare_data_for_visualization(df)
    
    # Generate plots
    print("Generating difficulty distribution plot...")
    plot_difficulty_distribution(train_df, val_df, test_df)
    
    print("Generating is_correct distribution plot...")
    plot_is_correct_distribution(train_df, val_df, test_df, all_df)
    
    print("Generating user level distribution plot...")
    plot_user_level_distribution(all_df)
    
    # Load difficulty comparison data and generate new plots
    print("Loading difficulty comparison data...")
    difficulty_df, metrics_df = load_difficulty_comparison_data()
    
    print("Generating difficulty comparison plot...")
    plot_difficulty_comparison(difficulty_df, metrics_df)
    
    print("Generating difficulty distributions plot...")
    plot_difficulty_distributions(difficulty_df)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main() 
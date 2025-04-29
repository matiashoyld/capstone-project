import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("09_holdout_irt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_holdout_prediction_matrix():
    """
    Load the holdout prediction matrix.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_path = os.path.join(current_dir, 'predictions', '08_holdout_probability_matrix.csv')
    
    try:
        logger.info(f"Loading holdout prediction matrix from {prediction_path}")
        df = pd.read_csv(prediction_path, index_col=0)
        logger.info(f"Loaded prediction matrix with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading prediction matrix: {e}")
        return None

def load_original_difficulties():
    """
    Load the original IRT difficulties from the holdout test set.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_path = os.path.join(current_dir, 'data', '07_holdout_test_questions.csv')
    
    try:
        logger.info(f"Loading original difficulties from {holdout_path}")
        df = pd.read_csv(holdout_path)
        # Create a mapping from question_id to difficulty
        difficulties = dict(zip(df['question_id'].astype(str), df['irt_difficulty']))
        logger.info(f"Loaded {len(difficulties)} original difficulties")
        return difficulties
    except Exception as e:
        logger.error(f"Error loading original difficulties: {e}")
        return {}

def compare_difficulties(estimated_df, original_difficulties):
    """
    Compare estimated difficulties with original difficulties.
    
    Args:
        estimated_df: DataFrame with question_id as index and difficulty as a column
        original_difficulties: Dictionary mapping question_id to original difficulty
        
    Returns:
        DataFrame with question_id, estimated_difficulty, and original_difficulty
    """
    # Create a comparison DataFrame
    comparison = []
    
    for q_id, row in estimated_df.iterrows():
        q_id_str = str(q_id)
        if q_id_str in original_difficulties:
            comparison.append({
                'question_id': q_id_str,
                'estimated_difficulty': row['difficulty'],
                'original_difficulty': original_difficulties[q_id_str],
                'discrimination': row['discrimination']
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Calculate correlation metrics
    pearson_corr, pearson_p = pearsonr(
        comparison_df['estimated_difficulty'], 
        comparison_df['original_difficulty']
    )
    
    spearman_corr, spearman_p = spearmanr(
        comparison_df['estimated_difficulty'], 
        comparison_df['original_difficulty']
    )
    
    mae = mean_absolute_error(
        comparison_df['original_difficulty'], 
        comparison_df['estimated_difficulty']
    )
    
    rmse = np.sqrt(mean_squared_error(
        comparison_df['original_difficulty'], 
        comparison_df['estimated_difficulty']
    ))
    
    logger.info(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    logger.info(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    
    # Update the comparison DataFrame with these metrics
    metrics = {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse
    }
    
    return comparison_df, metrics

def estimate_irt_parameters(response_matrix):
    """
    Estimate IRT parameters using 2PL model from the response matrix.
    
    Args:
        response_matrix: DataFrame with users as rows and questions as columns
        
    Returns:
        Tuple of (difficulties, discriminations, abilities)
    """
    logger.info("Estimating IRT parameters using PyTorch implementation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Convert response matrix to numpy array
    response_array = response_matrix.values
    n_users, n_questions = response_array.shape
    
    # Helper function to convert numpy arrays to PyTorch tensors
    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32, device=device)
    
    # Define the 2PL IRT model
    class IRT2PLModel(torch.nn.Module):
        def __init__(self, n_questions, n_users):
            super(IRT2PLModel, self).__init__()
            # Parameters
            self.difficulties = torch.nn.Parameter(torch.randn(n_questions, device=device))
            self.discriminations = torch.nn.Parameter(torch.ones(n_questions, device=device))
            self.abilities = torch.nn.Parameter(torch.randn(n_users, device=device))
        
        def forward(self):
            # Reshape for broadcasting
            theta = self.abilities.unsqueeze(1)  # [n_users, 1]
            b = self.difficulties.unsqueeze(0)   # [1, n_questions]
            a = self.discriminations.unsqueeze(0)  # [1, n_questions]
            
            # 2PL model: P(correct) = 1 / (1 + exp(-a(theta - b)))
            logits = a * (theta - b)
            probs = torch.sigmoid(logits)
            return probs
        
        def log_likelihood(self):
            # Get probabilities from forward pass
            probs = self.forward()
            
            # Convert to log probabilities
            log_probs = torch.log(probs)
            log_probs_incorrect = torch.log(1 - probs)
            
            # Get the log likelihood of observing the responses
            response_tensor = to_tensor(response_array)
            log_like = response_tensor * log_probs + (1 - response_tensor) * log_probs_incorrect
            
            # Return the negative log likelihood (for minimization)
            return -torch.sum(log_like)
    
    def estimate_parameters_torch():
        # Create model
        model = IRT2PLModel(n_questions, n_users).to(device)
        
        # Create optimizer with different learning rates for each parameter
        optimizer = optim.Adam([
            {'params': model.difficulties, 'lr': 0.01},
            {'params': model.discriminations, 'lr': 0.005},
            {'params': model.abilities, 'lr': 0.01}
        ])
        
        # Create learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        n_epochs = 500
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        pbar = tqdm(range(n_epochs), desc="Training IRT model")
        for epoch in pbar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.log_likelihood()
            
            # Backpropagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate scheduler
            scheduler.step(loss.item())
            
            # Print progress
            pbar.set_description(f"Training IRT model (loss: {loss.item():.4f})")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Extract parameters
        difficulties = model.difficulties.detach().cpu().numpy()
        discriminations = model.discriminations.detach().cpu().numpy()
        abilities = model.abilities.detach().cpu().numpy()
        
        return difficulties, discriminations, abilities
    
    # Estimate parameters
    difficulties, discriminations, abilities = estimate_parameters_torch()
    
    return difficulties, discriminations, abilities

def analyze_and_save_results(difficulties, discriminations, abilities, question_ids, user_ids, original_difficulties):
    """
    Analyze and save IRT parameter estimates.
    
    Args:
        difficulties: Array of difficulty estimates
        discriminations: Array of discrimination estimates
        abilities: Array of ability estimates
        question_ids: List of question IDs
        user_ids: List of user IDs
        original_difficulties: Dictionary mapping question_id to original difficulty
    """
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Save difficulty estimates
    difficulties_df = pd.DataFrame({
        'difficulty': difficulties,
        'discrimination': discriminations
    }, index=question_ids)
    
    difficulties_df.to_csv('results/09_holdout_irt_difficulties.csv')
    logger.info(f"Saved difficulty estimates to results/09_holdout_irt_difficulties.csv")
    
    # Save ability estimates
    abilities_df = pd.DataFrame({
        'ability': abilities
    }, index=user_ids)
    
    abilities_df.to_csv('results/09_holdout_irt_abilities.csv')
    logger.info(f"Saved ability estimates to results/09_holdout_irt_abilities.csv")
    
    # Compare with original difficulties
    comparison_df, metrics = compare_difficulties(difficulties_df, original_difficulties)
    comparison_df.to_csv('results/09_holdout_difficulty_comparison.csv')
    logger.info(f"Saved difficulty comparison to results/09_holdout_difficulty_comparison.csv")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv('results/09_holdout_irt_metrics.csv', index=False)
    logger.info(f"Saved metrics to results/09_holdout_irt_metrics.csv")
    
    # Plot comparison of estimated vs original difficulties
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color based on discrimination
    scatter = plt.scatter(
        comparison_df['original_difficulty'],
        comparison_df['estimated_difficulty'],
        c=comparison_df['discrimination'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Discrimination Parameter')
    
    # Add identity line (y=x)
    min_val = min(comparison_df['original_difficulty'].min(), comparison_df['estimated_difficulty'].min())
    max_val = max(comparison_df['original_difficulty'].max(), comparison_df['estimated_difficulty'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    # Add best fit line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        comparison_df['original_difficulty'],
        comparison_df['estimated_difficulty']
    )
    plt.plot(
        [min_val, max_val],
        [slope * min_val + intercept, slope * max_val + intercept],
        'b-',
        alpha=0.8
    )
    
    # Label the 5 points with largest discrepancy
    comparison_df['discrepancy'] = abs(comparison_df['estimated_difficulty'] - comparison_df['original_difficulty'])
    top_discrepancy = comparison_df.nlargest(5, 'discrepancy')
    
    for _, row in top_discrepancy.iterrows():
        plt.annotate(
            row['question_id'],
            (row['original_difficulty'], row['estimated_difficulty']),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7)
        )
    
    # Add correlation information to the plot
    plt.annotate(
        f"Pearson Correlation: {metrics['pearson_corr']:.4f}\n"
        f"Spearman Correlation: {metrics['spearman_corr']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    plt.title('Comparison of Original vs Estimated IRT Difficulties (Holdout Questions)')
    plt.xlabel('Original IRT Difficulty')
    plt.ylabel('Estimated IRT Difficulty')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/09_holdout_difficulty_comparison.png', dpi=300)
    logger.info(f"Saved difficulty comparison plot to figures/09_holdout_difficulty_comparison.png")
    plt.close()
    
    # Plot distribution of differences
    plt.figure(figsize=(10, 6))
    
    # Calculate differences
    comparison_df['difference'] = comparison_df['estimated_difficulty'] - comparison_df['original_difficulty']
    
    # Create histogram of differences
    sns.histplot(comparison_df['difference'], kde=True, bins=30)
    
    plt.title('Distribution of Differences Between Estimated and Original Difficulties')
    plt.xlabel('Difference (Estimated - Original)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Add vertical line at x=0
    plt.axvline(0, color='r', linestyle='--', alpha=0.8)
    
    # Add mean and median lines
    mean_diff = comparison_df['difference'].mean()
    median_diff = comparison_df['difference'].median()
    
    plt.axvline(mean_diff, color='g', linestyle='-', alpha=0.8, label=f'Mean: {mean_diff:.4f}')
    plt.axvline(median_diff, color='b', linestyle='-.', alpha=0.8, label=f'Median: {median_diff:.4f}')
    
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/09_holdout_difficulty_differences.png', dpi=300)
    logger.info(f"Saved difficulty differences plot to figures/09_holdout_difficulty_differences.png")
    plt.close()
    
    # Plot relationship between difficulty and discrimination
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(
        comparison_df['original_difficulty'],
        comparison_df['discrimination'],
        alpha=0.7,
        s=40
    )
    
    plt.title('Relationship Between Difficulty and Discrimination')
    plt.xlabel('Original Difficulty')
    plt.ylabel('Discrimination Parameter')
    plt.grid(alpha=0.3)
    
    # Add best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        comparison_df['original_difficulty'],
        comparison_df['discrimination']
    )
    
    x_vals = np.array([min_val, max_val])
    plt.plot(
        x_vals,
        slope * x_vals + intercept,
        'r--',
        alpha=0.8,
        label=f'r = {r_value:.4f}'
    )
    
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/09_holdout_difficulty_discrimination.png', dpi=300)
    logger.info(f"Saved difficulty-discrimination relationship plot to figures/09_holdout_difficulty_discrimination.png")
    plt.close()

def main():
    # Load prediction matrix
    response_matrix = load_holdout_prediction_matrix()
    if response_matrix is None:
        logger.error("Failed to load prediction matrix. Exiting.")
        return
    
    # Load original difficulties
    original_difficulties = load_original_difficulties()
    if not original_difficulties:
        logger.error("Failed to load original difficulties. Exiting.")
        return
    
    # Get question and user IDs
    question_ids = response_matrix.columns.tolist()
    user_ids = response_matrix.index.tolist()
    
    # Estimate IRT parameters
    difficulties, discriminations, abilities = estimate_irt_parameters(response_matrix)
    
    # Analyze and save results
    analyze_and_save_results(
        difficulties, 
        discriminations, 
        abilities, 
        question_ids, 
        user_ids,
        original_difficulties
    )
    
    logger.info("IRT analysis complete!")

if __name__ == "__main__":
    main() 
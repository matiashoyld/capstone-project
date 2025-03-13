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
        logging.FileHandler("06_neural_net_irt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_prediction_matrix():
    """
    Load the neural network prediction matrix.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_path = os.path.join(current_dir, 'predictions', '05_probability_matrix.csv')
    
    logger.info(f"Loading prediction matrix from {matrix_path}")
    
    try:
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        matrix_df.index = matrix_df.index.astype(int)
        matrix_df.columns = matrix_df.columns.astype(int)
        
        # Convert from probabilities to binary responses for IRT
        # We'll use the raw probabilities for the IRT model later
        
        logger.info(f"Loaded prediction matrix with shape: {matrix_df.shape}")
        return matrix_df
    except Exception as e:
        logger.error(f"Error loading prediction matrix: {e}")
        return None

def load_original_difficulties():
    """
    Load the original IRT difficulty values.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    difficulties_path = os.path.join(current_dir, 'data', 'question_difficulties_irt.csv')
    
    logger.info(f"Loading original difficulties from {difficulties_path}")
    
    try:
        difficulties_df = pd.read_csv(difficulties_path)
        logger.info(f"Loaded {len(difficulties_df)} original difficulty values")
        return difficulties_df
    except Exception as e:
        logger.error(f"Error loading original difficulties: {e}")
        return None

def compare_difficulties(estimated_df, original_df):
    """
    Compare estimated difficulties with original difficulties.
    
    Args:
        estimated_df: DataFrame with estimated difficulties
        original_df: DataFrame with original difficulties
    
    Returns:
        DataFrame with comparison and correlation metrics
    """
    # Merge the dataframes on question_id
    comparison_df = pd.merge(
        estimated_df,
        original_df[['question_id', 'irt_difficulty', 'original_difficulty']],
        on='question_id',
        how='inner'
    )
    
    logger.info(f"Comparing {len(comparison_df)} questions with both estimated and original difficulties")
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(comparison_df['difficulty'], comparison_df['irt_difficulty'])
    spearman_r, spearman_p = spearmanr(comparison_df['difficulty'], comparison_df['irt_difficulty'])
    
    # Calculate error metrics
    mae = mean_absolute_error(comparison_df['irt_difficulty'], comparison_df['difficulty'])
    rmse = np.sqrt(mean_squared_error(comparison_df['irt_difficulty'], comparison_df['difficulty']))
    
    logger.info(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4f})")
    logger.info(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4f})")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"Root Mean Square Error: {rmse:.4f}")
    
    # Find questions with largest discrepancies
    comparison_df['abs_diff'] = np.abs(comparison_df['difficulty'] - comparison_df['irt_difficulty'])
    comparison_df = comparison_df.sort_values('abs_diff', ascending=False)
    
    logger.info("Top 10 questions with largest discrepancies:")
    for i, row in comparison_df.head(10).iterrows():
        logger.info(f"Question {row['question_id']}: Estimated={row['difficulty']:.2f}, Original={row['irt_difficulty']:.2f}, Diff={row['abs_diff']:.2f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df['irt_difficulty'], comparison_df['difficulty'], alpha=0.6)
    plt.plot([-6, 6], [-6, 6], 'r--')  # Identity line
    plt.xlabel('Original IRT Difficulty')
    plt.ylabel('Estimated Difficulty (Neural Network)')
    plt.title(f'Original vs. Estimated Difficulty (r={pearson_r:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Add correlation and error info to plot
    text_info = f"Pearson r: {pearson_r:.4f}\nSpearman r: {spearman_r:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    plt.annotate(text_info, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(current_dir, 'figures')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, '06_difficulty_comparison.png'))
    logger.info(f"Saved difficulty comparison plot to {os.path.join(plot_dir, '06_difficulty_comparison.png')}")
    
    # Create histogram of differences
    plt.figure(figsize=(10, 6))
    plt.hist(comparison_df['abs_diff'], bins=30, alpha=0.7, color='blue')
    plt.axvline(mae, color='red', linestyle='--', label=f'MAE: {mae:.4f}')
    plt.xlabel('Absolute Difference (Estimated - Original)')
    plt.ylabel('Count')
    plt.title('Distribution of Absolute Differences in Difficulty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '06_difficulty_differences.png'))
    logger.info(f"Saved difficulty differences plot to {os.path.join(plot_dir, '06_difficulty_differences.png')}")
    
    # Save comparison dataframe
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(results_dir, '06_difficulty_comparison.csv'), index=False)
    logger.info(f"Saved difficulty comparison to {os.path.join(results_dir, '06_difficulty_comparison.csv')}")
    
    return comparison_df, {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mae': mae,
        'rmse': rmse
    }

def estimate_irt_parameters(response_matrix):
    """
    Estimate IRT parameters using a 2PL model from the neural network's probability matrix.
    
    Args:
        response_matrix: DataFrame with user responses (probabilities)
        
    Returns:
        Estimated difficulties, discriminations, and abilities
    """
    logger.info("Estimating IRT parameters using 2PL model...")
    
    # Extract indices for users and questions
    user_ids = response_matrix.index.values
    question_ids = response_matrix.columns.values
    
    n_users = len(user_ids)
    n_questions = len(question_ids)
    
    logger.info(f"Matrix dimensions: {n_users} users x {n_questions} questions")
    
    # Convert matrix to tensor
    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32)
    
    responses = to_tensor(response_matrix.values)
    
    # Define 2PL IRT model with PyTorch
    class IRT2PLModel(torch.nn.Module):
        def __init__(self, n_questions, n_users):
            super(IRT2PLModel, self).__init__()
            
            # Initialize parameters
            self.difficulties = torch.nn.Parameter(torch.randn(n_questions) * 0.1)
            self.discriminations = torch.nn.Parameter(torch.ones(n_questions) + torch.randn(n_questions) * 0.1)
            self.abilities = torch.nn.Parameter(torch.randn(n_users) * 0.1)
            
        def forward(self):
            # Reshape for broadcasting
            theta = self.abilities.unsqueeze(1)  # shape: (n_users, 1)
            b = self.difficulties.unsqueeze(0)   # shape: (1, n_questions)
            a = self.discriminations.unsqueeze(0)  # shape: (1, n_questions)
            
            # Calculate logits using 2PL model formula: a(θ - b)
            logits = a * (theta - b)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            return probs
        
        def log_likelihood(self):
            # Get probabilities from forward pass
            probs = self.forward()
            
            # Use the raw probabilities from neural network directly
            # We're calculating how likely the neural network's probabilities are under our IRT model
            
            # Binary cross entropy loss with raw probabilities
            log_likelihood = responses * torch.log(probs + 1e-8) + (1 - responses) * torch.log(1 - probs + 1e-8)
            
            # Sum log likelihood across all user-question pairs
            total_log_likelihood = torch.sum(log_likelihood)
            
            return total_log_likelihood
    
    def estimate_parameters_torch():
        # Create model
        model = IRT2PLModel(n_questions, n_users)
        
        # Use Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # Track progress
        pbar = tqdm(range(5000), desc="Training IRT model")
        log_liks = []
        
        for i in pbar:
            # Forward pass and calculate loss
            log_lik = model.log_likelihood()
            loss = -log_lik  # Negative log likelihood
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            log_liks.append(log_lik.item())
            
            if i % 100 == 0:
                pbar.set_description(f"Training IRT model (log-lik: {log_lik.item():.2f})")
                scheduler.step(log_lik.item())
            
            # Early stopping
            if i > 1000 and i % 100 == 0:
                if abs(log_liks[-1] - log_liks[-100]) < 1:
                    logger.info(f"Early stopping at iteration {i} (log-lik: {log_lik.item():.2f})")
                    break
        
        pbar.close()
        
        # Get final estimated parameters
        difficulties = model.difficulties.detach().numpy()
        discriminations = model.discriminations.detach().numpy()
        abilities = model.abilities.detach().numpy()
        
        # Ensure discriminations are positive
        discriminations = np.abs(discriminations)
        
        # Plot log likelihood over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(log_liks)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood During Training')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, 'figures')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, '06_log_likelihood.png'))
        logger.info(f"Saved log likelihood plot to {os.path.join(plot_dir, '06_log_likelihood.png')}")
        
        return difficulties, discriminations, abilities
    
    # Estimate parameters
    difficulties, discriminations, abilities = estimate_parameters_torch()
    
    logger.info(f"Estimated parameters: {len(difficulties)} difficulties, {len(discriminations)} discriminations, {len(abilities)} abilities")
    
    return difficulties, discriminations, abilities, question_ids, user_ids

def analyze_and_save_results(difficulties, discriminations, abilities, question_ids, user_ids):
    """
    Analyze and save the estimated IRT parameters.
    
    Args:
        difficulties: Array of difficulty parameters
        discriminations: Array of discrimination parameters
        abilities: Array of ability parameters
        question_ids: Array of question IDs
        user_ids: Array of user IDs
        
    Returns:
        DataFrames with estimated parameters
    """
    logger.info("Analyzing and saving IRT parameter estimates...")
    
    # Create DataFrames for estimated parameters
    difficulty_df = pd.DataFrame({
        'question_id': question_ids,
        'difficulty': difficulties,
        'discrimination': discriminations
    })
    
    ability_df = pd.DataFrame({
        'user_id': user_ids,
        'ability': abilities
    })
    
    # Calculate statistics
    logger.info(f"Difficulty statistics: Mean={difficulties.mean():.4f}, Std={difficulties.std():.4f}, Min={difficulties.min():.4f}, Max={difficulties.max():.4f}")
    logger.info(f"Discrimination statistics: Mean={discriminations.mean():.4f}, Std={discriminations.std():.4f}, Min={discriminations.min():.4f}, Max={discriminations.max():.4f}")
    logger.info(f"Ability statistics: Mean={abilities.mean():.4f}, Std={abilities.std():.4f}, Min={abilities.min():.4f}, Max={abilities.max():.4f}")
    
    # Create plots
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(current_dir, 'figures')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Difficulty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=30, alpha=0.7, color='blue')
    plt.axvline(difficulties.mean(), color='red', linestyle='--', label=f'Mean: {difficulties.mean():.4f}')
    plt.xlabel('Difficulty')
    plt.ylabel('Count')
    plt.title('Distribution of Estimated Difficulty Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '06_difficulty_distribution.png'))
    
    # Discrimination distribution
    plt.figure(figsize=(10, 6))
    plt.hist(discriminations, bins=30, alpha=0.7, color='green')
    plt.axvline(discriminations.mean(), color='red', linestyle='--', label=f'Mean: {discriminations.mean():.4f}')
    plt.xlabel('Discrimination')
    plt.ylabel('Count')
    plt.title('Distribution of Estimated Discrimination Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '06_discrimination_distribution.png'))
    
    # Ability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(abilities, bins=30, alpha=0.7, color='purple')
    plt.axvline(abilities.mean(), color='red', linestyle='--', label=f'Mean: {abilities.mean():.4f}')
    plt.xlabel('Ability')
    plt.ylabel('Count')
    plt.title('Distribution of Estimated Ability Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '06_ability_distribution.png'))
    
    # Scatter plot of difficulty vs. discrimination
    plt.figure(figsize=(10, 6))
    plt.scatter(difficulties, discriminations, alpha=0.6)
    plt.xlabel('Difficulty')
    plt.ylabel('Discrimination')
    plt.title('Difficulty vs. Discrimination Parameters')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '06_difficulty_vs_discrimination.png'))
    
    logger.info(f"Saved parameter distribution plots to {plot_dir}")
    
    # Save parameter estimates
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    difficulty_df.to_csv(os.path.join(results_dir, '06_estimated_difficulties.csv'), index=False)
    ability_df.to_csv(os.path.join(results_dir, '06_estimated_abilities.csv'), index=False)
    
    logger.info(f"Saved parameter estimates to {results_dir}")
    
    return difficulty_df, ability_df

def main():
    # Load prediction matrix
    response_matrix = load_prediction_matrix()
    if response_matrix is None:
        logger.error("Failed to load prediction matrix. Exiting.")
        return
    
    # Load original difficulties
    original_difficulties = load_original_difficulties()
    if original_difficulties is None:
        logger.error("Failed to load original difficulties. Exiting.")
        return
    
    # Estimate IRT parameters
    difficulties, discriminations, abilities, question_ids, user_ids = estimate_irt_parameters(response_matrix)
    
    # Analyze and save results
    difficulty_df, ability_df = analyze_and_save_results(
        difficulties, discriminations, abilities, question_ids, user_ids
    )
    
    # Compare with original difficulties
    comparison_df, metrics = compare_difficulties(difficulty_df, original_difficulties)
    
    logger.info("IRT parameter estimation from neural network predictions completed.")
    logger.info(f"Correlation with original difficulties: Pearson={metrics['pearson_r']:.4f}, Spearman={metrics['spearman_r']:.4f}")
    logger.info(f"Error metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

if __name__ == "__main__":
    main() 
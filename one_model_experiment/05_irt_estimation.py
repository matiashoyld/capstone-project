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
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("05_irt_estimation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_prediction_matrix():
    """
    Load the binary prediction matrix created by 04_predictions.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_path = os.path.join(current_dir, 'predictions', '04_binary_prediction_matrix.csv')
    logger.info(f"Loading prediction matrix from {matrix_path}")
    
    # Load the CSV file
    matrix_df = pd.read_csv(matrix_path, index_col=0)
    
    # Extract question_ids and user_ids
    question_ids = matrix_df.index.tolist()
    user_ids = [int(col) for col in matrix_df.columns]
    
    # Convert to numpy array
    matrix = matrix_df.values
    
    logger.info(f"Loaded matrix with shape {matrix.shape} ({len(question_ids)} questions, {len(user_ids)} users)")
    return matrix, question_ids, user_ids

def load_original_difficulties():
    """
    Load the original IRT difficulties from the dataset
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    logger.info(f"Loading original difficulties from {data_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Extract question IDs and IRT difficulties
        question_df = df[['question_id', 'irt_difficulty']].drop_duplicates()
        
        # Check if we have irt_difficulty column
        if 'irt_difficulty' not in question_df.columns:
            logger.warning("Column 'irt_difficulty' not found in the data")
            return None
        
        logger.info(f"Loaded original difficulties for {len(question_df)} questions")
        return question_df
    
    except Exception as e:
        logger.error(f"Error loading original difficulties: {e}")
        return None

def compare_difficulties(estimated_df, original_df):
    """
    Compare estimated difficulties with original IRT difficulties
    
    Args:
        estimated_df: DataFrame with estimated difficulties
        original_df: DataFrame with original IRT difficulties
    """
    if original_df is None:
        logger.warning("No original difficulties available for comparison")
        return
    
    # Merge dataframes on question_id
    merged_df = pd.merge(
        estimated_df, 
        original_df,
        on='question_id',
        how='inner'
    )
    
    # Check if we have any matches
    if len(merged_df) == 0:
        logger.warning("No matching question IDs found between estimated and original difficulties")
        return
    
    logger.info(f"Comparing difficulties for {len(merged_df)} questions")
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(merged_df['difficulty'], merged_df['irt_difficulty'])
    spearman_corr, spearman_p = spearmanr(merged_df['difficulty'], merged_df['irt_difficulty'])
    
    logger.info(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    logger.info(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Create directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(current_dir, 'figures')
    output_dir = os.path.join(current_dir, 'irt_results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['irt_difficulty'], merged_df['difficulty'], alpha=0.5)
    plt.xlabel('Original IRT Difficulty')
    plt.ylabel('Estimated Difficulty')
    plt.title(f'Original vs Estimated Difficulty (Pearson r={pearson_corr:.4f})')
    
    # Add identity line (x=y)
    min_val = min(merged_df['irt_difficulty'].min(), merged_df['difficulty'].min())
    max_val = max(merged_df['irt_difficulty'].max(), merged_df['difficulty'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_difficulty_comparison.png'))
    logger.info("Saved difficulty comparison plot")
    
    # Save comparison results
    merged_df.to_csv(os.path.join(output_dir, '05_difficulty_comparison.csv'), index=False)
    logger.info("Saved difficulty comparison data")
    
    # Create a histogram of the differences
    plt.figure(figsize=(10, 6))
    differences = merged_df['difficulty'] - merged_df['irt_difficulty']
    plt.hist(differences, bins=30)
    plt.xlabel('Estimated - Original Difficulty')
    plt.ylabel('Count')
    plt.title('Distribution of Difficulty Differences')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_difficulty_differences.png'))
    logger.info("Saved difficulty differences histogram")
    
    # Calculate error metrics
    mae = np.abs(differences).mean()
    rmse = np.sqrt((differences ** 2).mean())
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    
    # Find questions with largest differences
    merged_df['difference'] = differences
    merged_df['abs_difference'] = np.abs(differences)
    
    logger.info("\nQuestions with largest overestimation (Estimated > Original):")
    logger.info(merged_df.sort_values('difference', ascending=False).head(10)[['question_id', 'difficulty', 'irt_difficulty', 'difference']])
    
    logger.info("\nQuestions with largest underestimation (Original > Estimated):")
    logger.info(merged_df.sort_values('difference').head(10)[['question_id', 'difficulty', 'irt_difficulty', 'difference']])
    
    return merged_df

def estimate_irt_parameters(response_matrix):
    """
    Estimate IRT parameters using a 2PL model with PyTorch
    
    Args:
        response_matrix: Binary matrix of responses (questions x users)
        
    Returns:
        difficulties: Difficulty parameters for questions
        discriminations: Discrimination parameters for questions
        abilities: Ability parameters for users
    """
    logger.info("Starting IRT parameter estimation (2PL model)")
    
    # Get matrix dimensions
    n_questions, n_users = response_matrix.shape
    
    # Convert to PyTorch tensors
    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32, device=device)
    
    responses = to_tensor(response_matrix)
    
    # Create mask for valid responses (no NaN values in our case)
    valid_mask = ~torch.isnan(responses)
    
    # Define the 2PL IRT model
    class IRT2PLModel(torch.nn.Module):
        def __init__(self, n_questions, n_users):
            super(IRT2PLModel, self).__init__()
            # Initialize parameters with appropriate priors
            # Difficulty parameters (normally distributed around 0)
            self.difficulties = torch.nn.Parameter(torch.randn(n_questions) * 0.1)
            
            # Discrimination parameters (log-normally distributed, positive)
            self.discriminations_raw = torch.nn.Parameter(torch.randn(n_questions) * 0.1)
            
            # Ability parameters (normally distributed around 0)
            self.abilities = torch.nn.Parameter(torch.randn(n_users) * 0.1)
            
        def forward(self):
            # Reshape for broadcasting
            abilities = self.abilities.unsqueeze(0)  # [1, n_users]
            difficulties = self.difficulties.unsqueeze(1)  # [n_questions, 1]
            
            # Apply sigmoid function to ensure discriminations are positive
            discriminations = torch.nn.functional.softplus(self.discriminations_raw).unsqueeze(1)  # [n_questions, 1]
            
            # Calculate the probability of correct response using 2PL model
            # P(correct) = 1 / (1 + exp(-a(θ - b)))
            logits = discriminations * (abilities - difficulties)
            probs = torch.sigmoid(logits)
            
            return probs
        
        def log_likelihood(self):
            # Get probabilities from forward pass
            probs = self.forward()
            
            # Constrain discriminations to reasonable bounds using softplus
            # This is a soft constraint through penalty
            discriminations = torch.nn.functional.softplus(self.discriminations_raw)
            
            # Calculate log likelihood for valid responses
            log_probs = torch.where(
                responses == 1,
                torch.log(probs + 1e-10),
                torch.log(1 - probs + 1e-10)
            )
            
            # Only consider valid (non-NaN) responses
            log_likelihood = torch.sum(log_probs * valid_mask)
            
            # Add regularization to prevent extreme parameter values
            # L2 regularization for abilities
            ability_reg = 0.01 * torch.sum(self.abilities ** 2)
            # L2 regularization for difficulties
            difficulty_reg = 0.01 * torch.sum(self.difficulties ** 2)
            # L2 regularization for discriminations with stronger weight
            discrimination_reg = 0.1 * torch.sum((discriminations - 1.0) ** 2)
            
            # Return negative log likelihood (for minimization) with regularization
            return -log_likelihood + ability_reg + difficulty_reg + discrimination_reg
    
    # Function to perform the actual estimation
    def estimate_parameters_torch():
        # Create model
        model = IRT2PLModel(n_questions, n_users).to(device)
        
        # Use Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Use learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Number of epochs
        n_epochs = 500
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 20
        counter = 0
        
        # Training loop
        losses = []
        
        logger.info(f"Training for {n_epochs} epochs with early stopping")
        progress_bar = tqdm(range(n_epochs), desc="Training IRT model")
        
        for epoch in progress_bar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate loss
            loss = model.log_likelihood()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update scheduler
            scheduler.step(loss.item())
            
            # Add loss to list
            losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}/{n_epochs}")
                break
                
        # Retrieve and return parameters
        with torch.no_grad():
            difficulties = model.difficulties.cpu().numpy()
            discriminations = torch.nn.functional.softplus(model.discriminations_raw).cpu().numpy()
            abilities = model.abilities.cpu().numpy()
            
        return difficulties, discriminations, abilities, losses
    
    # Perform estimation
    try:
        difficulties, discriminations, abilities, losses = estimate_parameters_torch()
        logger.info("IRT parameter estimation completed successfully")
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood + Regularization')
        plt.title('IRT Model Training Loss')
        plt.grid(True, alpha=0.3)
        
        # Create directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(current_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        plt.savefig(os.path.join(fig_dir, '05_irt_loss_curve.png'))
        logger.info("Saved loss curve figure")
        
        return difficulties, discriminations, abilities
    
    except Exception as e:
        logger.error(f"Error in IRT parameter estimation: {e}")
        return None, None, None

def analyze_and_save_results(difficulties, discriminations, abilities, question_ids, user_ids):
    """
    Analyze, visualize and save the IRT parameter estimation results
    """
    # Create DataFrames for results
    difficulty_df = pd.DataFrame({
        'question_id': question_ids,
        'difficulty': difficulties,
        'discrimination': discriminations
    })
    
    ability_df = pd.DataFrame({
        'user_id': user_ids,
        'ability': abilities
    })
    
    # Create directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'irt_results')
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(current_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Sort dataframes
    difficulty_df = difficulty_df.sort_values('difficulty')
    ability_df = ability_df.sort_values('ability')
    
    # Save results
    difficulty_df.to_csv(os.path.join(output_dir, '05_question_parameters.csv'), index=False)
    ability_df.to_csv(os.path.join(output_dir, '05_user_abilities.csv'), index=False)
    logger.info("Saved IRT parameters to CSV files")
    
    # Visualize difficulty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=30)
    plt.xlabel('Question Difficulty')
    plt.ylabel('Count')
    plt.title('Distribution of Question Difficulties')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_question_difficulties.png'))
    
    # Visualize discrimination distribution
    plt.figure(figsize=(10, 6))
    plt.hist(discriminations, bins=30)
    plt.xlabel('Question Discrimination')
    plt.ylabel('Count')
    plt.title('Distribution of Question Discriminations')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_question_discriminations.png'))
    
    # Visualize ability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(abilities, bins=30)
    plt.xlabel('User Ability')
    plt.ylabel('Count')
    plt.title('Distribution of User Abilities')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_user_abilities.png'))
    
    # Visualize difficulty vs discrimination
    plt.figure(figsize=(10, 6))
    plt.scatter(difficulties, discriminations, alpha=0.5)
    plt.xlabel('Difficulty')
    plt.ylabel('Discrimination')
    plt.title('Question Difficulty vs Discrimination')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, '05_difficulty_vs_discrimination.png'))
    
    logger.info("Saved all visualization figures")
    
    # Basic statistics
    logger.info("\nQuestion Parameters Statistics:")
    logger.info(f"Difficulty - Mean: {difficulties.mean():.4f}, Std: {difficulties.std():.4f}, Min: {difficulties.min():.4f}, Max: {difficulties.max():.4f}")
    logger.info(f"Discrimination - Mean: {discriminations.mean():.4f}, Std: {discriminations.std():.4f}, Min: {discriminations.min():.4f}, Max: {discriminations.max():.4f}")
    
    logger.info("\nUser Ability Statistics:")
    logger.info(f"Ability - Mean: {abilities.mean():.4f}, Std: {abilities.std():.4f}, Min: {abilities.min():.4f}, Max: {abilities.max():.4f}")
    
    return difficulty_df, ability_df

def main():
    # Load prediction matrix
    response_matrix, question_ids, user_ids = load_prediction_matrix()
    
    # Load original difficulties
    original_difficulties = load_original_difficulties()
    
    # Estimate IRT parameters
    difficulties, discriminations, abilities = estimate_irt_parameters(response_matrix)
    
    if difficulties is not None:
        # Analyze and save results
        difficulty_df, ability_df = analyze_and_save_results(
            difficulties, discriminations, abilities, question_ids, user_ids
        )
        
        # Compare estimated difficulties with original difficulties
        if original_difficulties is not None:
            comparison_df = compare_difficulties(difficulty_df, original_difficulties)
        
        # Print top 10 easiest and most difficult questions
        logger.info("\nTop 10 Easiest Questions:")
        logger.info(difficulty_df.sort_values('difficulty').head(10))
        
        logger.info("\nTop 10 Most Difficult Questions:")
        logger.info(difficulty_df.sort_values('difficulty', ascending=False).head(10))
        
        # Print top 10 most and least discriminating questions
        logger.info("\nTop 10 Most Discriminating Questions:")
        logger.info(difficulty_df.sort_values('discrimination', ascending=False).head(10))
        
        # Print top 10 highest and lowest ability users
        logger.info("\nTop 10 Highest Ability Users:")
        logger.info(ability_df.sort_values('ability', ascending=False).head(10))
        
        logger.info("\nTop 10 Lowest Ability Users:")
        logger.info(ability_df.sort_values('ability').head(10))
        
        logger.info("IRT analysis completed successfully")
    else:
        logger.error("IRT analysis failed")

if __name__ == "__main__":
    main() 
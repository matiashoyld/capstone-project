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
        logging.FileHandler("10_holdout_irt_all_users.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load the processed data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df

def load_holdout_questions():
    """
    Load the 10% holdout test set questions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_path = os.path.join(current_dir, 'data', '07_holdout_test_questions.csv')
    logger.info(f"Loading holdout questions from {holdout_path}")
    
    holdout_df = pd.read_csv(holdout_path)
    holdout_question_ids = holdout_df['question_id'].astype(str).tolist()
    
    logger.info(f"Loaded {len(holdout_question_ids)} holdout question IDs")
    return holdout_question_ids, holdout_df

def load_model_preprocessors():
    """
    Load the trained neural network model preprocessors.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessors_path = os.path.join(current_dir, 'models', '07_neural_net_holdout_preprocessors.pkl')
    
    try:
        import pickle
        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
        logger.info(f"Loaded preprocessors from {preprocessors_path}")
        return preprocessors
    except Exception as e:
        logger.error(f"Error loading preprocessors: {e}")
        return None

def load_embeddings():
    """
    Load question embeddings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    
    try:
        import pickle
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {embeddings_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return None

def generate_predictions(df, preprocessors, holdout_question_ids, model_path):
    """
    Generate predictions for all users on the holdout questions.
    
    Args:
        df: DataFrame with all data
        preprocessors: Dictionary of preprocessors
        holdout_question_ids: List of holdout question IDs
        model_path: Path to the trained model
        
    Returns:
        DataFrame with predictions, with users as rows and questions as columns
    """
    from tensorflow.keras.models import load_model
    
    # Load the model
    model = load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Get necessary preprocessors
    user_id_to_index = preprocessors['user_id_to_index']
    scaler = preprocessors['scaler']
    numerical_features = preprocessors['numerical_features']
    
    # Create a filtered dataframe with only holdout questions
    holdout_df = df[df['question_id'].astype(str).isin(holdout_question_ids)].copy()
    
    # Get unique users
    unique_users = df['user_id'].unique()
    logger.info(f"Found {len(unique_users)} unique users in the dataset")
    
    # Get unique question info (we only need one row per question)
    question_info = df.drop_duplicates('question_id').set_index('question_id')
    
    # Check if embeddings are needed
    embeddings = None
    question_embeddings = {}
    
    if len(model.inputs) > 2:  # Model uses embeddings
        logger.info("Model uses embeddings, loading...")
        embeddings = load_embeddings()
        
        # Create mapping from question_id to embedding
        if embeddings is not None and 'formatted_embeddings' in embeddings:
            for q_id, emb in zip(embeddings['question_ids'], embeddings['formatted_embeddings']):
                question_embeddings[str(q_id)] = emb
            embedding_dim = embeddings['formatted_embeddings'].shape[1]
            logger.info(f"Created embeddings mapping with dimension {embedding_dim}")
        else:
            logger.warning("Could not create question embeddings mapping")
    
    # Create prediction matrix
    predictions = {}
    batch_size = 1000  # Process users in batches to avoid memory issues
    
    # Process users in batches
    for i in tqdm(range(0, len(unique_users), batch_size), desc="Generating predictions"):
        batch_users = unique_users[i:i+batch_size]
        
        # For each user in the batch
        for user_id in batch_users:
            # Skip users not in the mapping
            if user_id not in user_id_to_index:
                continue
                
            # Convert user ID to index
            user_idx = user_id_to_index[user_id]
            
            # Prepare inputs for all questions for this user
            user_batch = np.array([user_idx] * len(holdout_question_ids))
            numerical_batch = np.zeros((len(holdout_question_ids), len(numerical_features)))
            embedding_batch = None
            
            if embeddings is not None:
                embedding_batch = np.zeros((len(holdout_question_ids), embedding_dim))
            
            # Fill features for each question
            for j, q_id in enumerate(holdout_question_ids):
                q_id_numeric = int(q_id) if q_id.isdigit() else q_id
                
                # Get numerical features
                if q_id_numeric in question_info.index:
                    features = question_info.loc[q_id_numeric, numerical_features].copy()
                    for col in numerical_features:
                        if pd.isna(features[col]):
                            features[col] = 0
                    numerical_batch[j] = features.values
                
                # Get embedding if available
                if embedding_batch is not None and q_id in question_embeddings:
                    embedding_batch[j] = question_embeddings[q_id]
            
            # Scale numerical features
            numerical_batch = scaler.transform(numerical_batch)
            
            # Make predictions
            inputs = [user_batch, numerical_batch]
            if embedding_batch is not None:
                inputs.append(embedding_batch)
            
            probs = model.predict(inputs, verbose=0).flatten()
            
            # Store predictions for this user
            predictions[user_id] = probs
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(predictions).T
    prediction_df.columns = holdout_question_ids
    
    logger.info(f"Generated predictions with shape {prediction_df.shape}")
    return prediction_df

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
        'rmse': rmse,
        'num_questions': len(comparison_df)
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
    
    difficulties_df.to_csv('results/10_holdout_irt_all_users_difficulties.csv')
    logger.info(f"Saved difficulty estimates to results/10_holdout_irt_all_users_difficulties.csv")
    
    # Save ability estimates
    abilities_df = pd.DataFrame({
        'ability': abilities
    }, index=user_ids)
    
    abilities_df.to_csv('results/10_holdout_irt_all_users_abilities.csv')
    logger.info(f"Saved ability estimates to results/10_holdout_irt_all_users_abilities.csv")
    
    # Compare with original difficulties
    comparison_df, metrics = compare_difficulties(difficulties_df, original_difficulties)
    comparison_df.to_csv('results/10_holdout_all_users_difficulty_comparison.csv')
    logger.info(f"Saved difficulty comparison to results/10_holdout_all_users_difficulty_comparison.csv")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv('results/10_holdout_irt_all_users_metrics.csv', index=False)
    logger.info(f"Saved metrics to results/10_holdout_irt_all_users_metrics.csv")
    
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
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"N = {metrics['num_questions']} questions",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    plt.title('Comparison of Original vs Estimated IRT Difficulties (All Users, Holdout Questions)')
    plt.xlabel('Original IRT Difficulty')
    plt.ylabel('Estimated IRT Difficulty')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/10_holdout_all_users_difficulty_comparison.png', dpi=300)
    logger.info(f"Saved difficulty comparison plot to figures/10_holdout_all_users_difficulty_comparison.png")
    plt.close()
    
    # Plot distribution of differences
    plt.figure(figsize=(10, 6))
    
    # Calculate differences
    comparison_df['difference'] = comparison_df['estimated_difficulty'] - comparison_df['original_difficulty']
    
    # Create histogram of differences
    sns.histplot(comparison_df['difference'], kde=True, bins=30)
    
    plt.title('Distribution of Differences Between Estimated and Original Difficulties (All Users)')
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
    plt.savefig('figures/10_holdout_all_users_difficulty_differences.png', dpi=300)
    logger.info(f"Saved difficulty differences plot to figures/10_holdout_all_users_difficulty_differences.png")
    plt.close()
    
    # Compare with previous results (from top 100 users)
    try:
        prev_metrics = pd.read_csv('results/09_holdout_irt_metrics.csv')
        
        # Create comparison table
        comparison_table = pd.DataFrame({
            'Metric': ['Pearson Correlation', 'Spearman Correlation', 'MAE', 'RMSE'],
            'Top 100 Users': [
                prev_metrics['pearson_corr'].iloc[0],
                prev_metrics['spearman_corr'].iloc[0],
                prev_metrics['mae'].iloc[0],
                prev_metrics['rmse'].iloc[0]
            ],
            'All Users': [
                metrics['pearson_corr'],
                metrics['spearman_corr'],
                metrics['mae'],
                metrics['rmse']
            ]
        })
        
        # Save comparison table
        comparison_table.to_csv('results/10_top100_vs_all_users_comparison.csv', index=False)
        logger.info(f"Saved metrics comparison to results/10_top100_vs_all_users_comparison.csv")
        
        # Create bar chart comparing the metrics
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = ['Pearson Correlation', 'Spearman Correlation']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        top100_values = [prev_metrics['pearson_corr'].iloc[0], prev_metrics['spearman_corr'].iloc[0]]
        all_values = [metrics['pearson_corr'], metrics['spearman_corr']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, top100_values, width, label='Top 100 Users')
        ax.bar(x + width/2, all_values, width, label='All Users')
        
        ax.set_ylabel('Correlation Value')
        ax.set_title('Difficulty Estimation Performance: Top 100 Users vs All Users')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('figures/10_correlation_comparison.png', dpi=300)
        logger.info(f"Saved correlation comparison plot to figures/10_correlation_comparison.png")
        plt.close()
        
        # Create bar chart for error metrics
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = ['MAE', 'RMSE']
        x = np.arange(len(metrics_to_plot))
        
        top100_values = [prev_metrics['mae'].iloc[0], prev_metrics['rmse'].iloc[0]]
        all_values = [metrics['mae'], metrics['rmse']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, top100_values, width, label='Top 100 Users')
        ax.bar(x + width/2, all_values, width, label='All Users')
        
        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics: Top 100 Users vs All Users')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('figures/10_error_comparison.png', dpi=300)
        logger.info(f"Saved error comparison plot to figures/10_error_comparison.png")
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not compare with previous results: {e}")

def save_predictions(response_matrix):
    """
    Save the response matrix to a CSV file.
    
    Args:
        response_matrix: DataFrame with users as rows and questions as columns
    """
    # Create directory for predictions
    os.makedirs('predictions', exist_ok=True)
    
    # Save response matrix
    prediction_path = os.path.join('predictions', '10_holdout_all_users_probability_matrix.csv')
    response_matrix.to_csv(prediction_path)
    logger.info(f"Saved predictions to {prediction_path}")

def main():
    # Load data
    df = load_data()
    
    # Load holdout questions
    holdout_question_ids, holdout_df = load_holdout_questions()
    
    # Create a mapping from question_id to difficulty
    original_difficulties = dict(zip(holdout_df['question_id'].astype(str), holdout_df['irt_difficulty']))
    
    # Load preprocessors
    preprocessors = load_model_preprocessors()
    if preprocessors is None:
        logger.error("Failed to load preprocessors. Exiting.")
        return
    
    # Check if predictions already exist
    prediction_path = os.path.join('predictions', '10_holdout_all_users_probability_matrix.csv')
    if os.path.exists(prediction_path):
        logger.info(f"Loading existing predictions from {prediction_path}")
        response_matrix = pd.read_csv(prediction_path, index_col=0)
    else:
        # Generate predictions for all users on holdout questions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', '07_neural_net_holdout.h5')
        response_matrix = generate_predictions(df, preprocessors, holdout_question_ids, model_path)
        
        # Save predictions
        save_predictions(response_matrix)
    
    # Get question and user IDs
    question_ids = response_matrix.columns.tolist()
    user_ids = response_matrix.index.tolist()
    
    logger.info(f"Response matrix has {len(user_ids)} users and {len(question_ids)} questions")
    
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
    
    logger.info("IRT analysis with all users complete!")

if __name__ == "__main__":
    main() 
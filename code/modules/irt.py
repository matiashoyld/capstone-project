import pandas as pd
import logging
from tqdm import tqdm
import numpy as np
# Replace PyTorch imports with TensorFlow
# import tensorflow as tf  # Keep TF for 1PL for now, add PyTorch
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging (optional, but good practice)
logger = logging.getLogger(__name__)
# Removed basicConfig to avoid conflict if main script configures root logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def estimate_irt_1pl_difficulty(
    response_df: pd.DataFrame,
    question_col: str = 'question_id',
    user_col: str = 'user_id',
    correctness_col: str = 'correctness',
    n_epochs: int = 100,
    lr: float = 0.05,
    patience: int = 10,
    min_delta: float = 1e-4
) -> pd.DataFrame:
    """
    Estimates IRT 1PL (Rasch model) difficulty parameters for questions using TensorFlow.

    Args:
        response_df: DataFrame containing response data.
        question_col: Name of the column containing question identifiers.
        user_col: Name of the column containing user identifiers.
        correctness_col: Name of the column containing correctness (True/1 for correct, False/0 for incorrect).
        n_epochs: Maximum number of training epochs.
        lr: Learning rate for the optimizer.
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in loss to qualify as improvement for early stopping.

    Returns:
        DataFrame with question_id and estimated difficulty.
    """
    logger.info("Starting IRT 1PL difficulty estimation using TensorFlow.")

    # --- Data Preprocessing (Same as before) ---
    logger.info("Preprocessing data...")
    response_df = response_df[[user_col, question_col, correctness_col]].copy()
    response_df[correctness_col] = response_df[correctness_col].astype(float)

    user_ids = response_df[user_col].unique()
    question_ids = response_df[question_col].unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    question_map = {qid: i for i, qid in enumerate(question_ids)}
    n_users = len(user_ids)
    n_questions = len(question_ids)
    logger.info(f"Found {n_users} unique users and {n_questions} unique questions.")

    response_df['user_idx'] = response_df[user_col].map(user_map)
    response_df['question_idx'] = response_df[question_col].map(question_map)

    # Convert to TensorFlow Tensors
    user_indices = tf.constant(response_df['user_idx'].values, dtype=tf.int32)
    question_indices = tf.constant(response_df['question_idx'].values, dtype=tf.int32)
    correctness = tf.constant(response_df[correctness_col].values, dtype=tf.float32)

    # --- 1PL Model Parameters (using tf.Variable) ---
    abilities = tf.Variable(tf.random.normal([n_users], stddev=0.1), name="abilities")
    difficulties = tf.Variable(tf.random.normal([n_questions], stddev=0.1), name="difficulties")

    # --- Training Setup ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function # Compile graph for speed
    def train_step():
        with tf.GradientTape() as tape:
            # Gather corresponding abilities and difficulties
            theta = tf.gather(abilities, user_indices)
            beta = tf.gather(difficulties, question_indices)

            # Calculate logits (theta - beta)
            logits = theta - beta

            # Calculate loss using sigmoid cross entropy
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=correctness, logits=logits))

            # Optional: Add L2 regularization
            # l2_loss = 0.01 * (tf.reduce_sum(tf.square(abilities)) + tf.reduce_sum(tf.square(difficulties)))
            # total_loss = loss + l2_loss
            total_loss = loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, [abilities, difficulties])
        optimizer.apply_gradients(zip(gradients, [abilities, difficulties]))
        return total_loss

    # --- Training Loop ---
    logger.info(f"Training for up to {n_epochs} epochs with Adam optimizer (lr={lr}).")
    best_loss = float('inf')
    epochs_no_improve = 0
    losses = []

    progress_bar = tqdm(range(n_epochs), desc="Training IRT 1PL (TF)")
    for epoch in progress_bar:
        current_loss = train_step()
        current_loss_numpy = current_loss.numpy()
        losses.append(current_loss_numpy)
        progress_bar.set_postfix({"loss": f"{current_loss_numpy:.4f}"})

        # Early stopping check
        if current_loss_numpy < best_loss - min_delta:
            best_loss = current_loss_numpy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    logger.info(f"Training finished. Final loss: {losses[-1]:.4f}")

    # --- Extract Results ---
    estimated_difficulties = difficulties.numpy()
    # Center difficulties by subtracting the mean
    estimated_difficulties = estimated_difficulties - np.mean(estimated_difficulties)
    # estimated_abilities = abilities.numpy()

    inv_question_map = {v: k for k, v in question_map.items()}
    difficulty_df = pd.DataFrame({
        'question_idx': range(n_questions),
        'difficulty': estimated_difficulties # Use centered difficulties
    })
    difficulty_df[question_col] = difficulty_df['question_idx'].map(inv_question_map)

    result_df = difficulty_df[[question_col, 'difficulty']].sort_values(by=question_col).reset_index(drop=True)

    logger.info("IRT 1PL difficulty estimation (TensorFlow) completed.")
    return result_df

# --- New 2PL PyTorch Function ---
class IRT2PLModel(nn.Module):
    def __init__(self, n_questions, n_users):
        super().__init__()
        # Initialize parameters
        self.difficulty = nn.Parameter(torch.randn(n_questions) * 0.1)
        # Initialize log_discrimination to keep discrimination positive after exp()
        self.log_discrimination = nn.Parameter(torch.randn(n_questions) * 0.1) 
        self.ability = nn.Parameter(torch.randn(n_users) * 0.1)

    def forward(self, user_indices, question_indices):
        # Get parameters for the specific interactions
        theta = self.ability[user_indices]
        b = self.difficulty[question_indices]
        a = torch.exp(self.log_discrimination[question_indices]) # Ensure discrimination is positive
        
        # Calculate logits: a * (theta - b)
        logits = a * (theta - b)
        return torch.sigmoid(logits)

def estimate_irt_2pl_params(
    response_df: pd.DataFrame,
    question_col: str = 'question_id',
    user_col: str = 'user_id',
    correctness_col: str = 'correctness',
    n_epochs: int = 200, 
    lr: float = 0.01,    
    patience: int = 15,  
    min_delta: float = 1e-5, 
    reg_lambda: float = 0.0 # Changed default to 0.0
) -> pd.DataFrame:
    """
    Estimates IRT 2PL parameters (difficulty, discrimination, ability) 
    using PyTorch with optional L2 regularization.

    Args:
        response_df: DataFrame with user_id, question_id, correctness (0 or 1).
        ...
        reg_lambda: Strength of L2 regularization. Set to 0 to disable.

    Returns:
        DataFrame with question_id, estimated difficulty, and estimated discrimination.
    """
    logger.info("Starting IRT 2PL parameter estimation using PyTorch.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Preprocessing ---
    logger.info("Preprocessing data for PyTorch...")
    response_df = response_df[[user_col, question_col, correctness_col]].copy()
    response_df[correctness_col] = response_df[correctness_col].astype(float)

    user_ids = response_df[user_col].unique()
    question_ids = response_df[question_col].unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    question_map = {qid: i for i, qid in enumerate(question_ids)}
    n_users = len(user_ids)
    n_questions = len(question_ids)
    logger.info(f"Found {n_users} unique users and {n_questions} unique questions.")

    # Map IDs to indices
    user_indices = torch.tensor(response_df[user_col].map(user_map).values, dtype=torch.long).to(device)
    question_indices = torch.tensor(response_df[question_col].map(question_map).values, dtype=torch.long).to(device)
    correctness = torch.tensor(response_df[correctness_col].values, dtype=torch.float32).to(device)

    # --- Model, Loss, Optimizer ---
    model = IRT2PLModel(n_questions, n_users).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='mean') 

    # --- Training Loop ---
    logger.info(f"Training for up to {n_epochs} epochs with Adam optimizer (lr={lr}) and L2 reg (lambda={reg_lambda}).")
    best_loss = float('inf')
    epochs_no_improve = 0
    losses = []
    
    progress_bar = tqdm(range(n_epochs), desc="Training IRT 2PL (PyTorch)")
    for epoch in progress_bar:
        model.train() 
        optimizer.zero_grad() 
        
        probabilities = model(user_indices, question_indices)
        
        bce_loss = criterion(probabilities, correctness)
        
        # Calculate L2 regularization penalty only if reg_lambda > 0
        # l2_reg = torch.tensor(0.).to(device)
        # if reg_lambda > 0:
        #     for param in model.parameters():
        #         l2_reg += torch.norm(param, p=2) ** 2
        #     total_loss = bce_loss + reg_lambda * l2_reg
        # else:
        #     total_loss = bce_loss
        total_loss = bce_loss # Directly use BCE loss if no regularization
        if reg_lambda > 0:
             l2_reg = torch.tensor(0.).to(device)
             for param in model.parameters():
                 l2_reg += torch.norm(param, p=2) ** 2
             total_loss = total_loss + reg_lambda * l2_reg # Add regularization if specified
        
        total_loss.backward()
        optimizer.step()
        
        current_loss_numpy = total_loss.item()
        losses.append(current_loss_numpy)
        progress_bar.set_postfix({"loss": f"{current_loss_numpy:.4f}", "bce": f"{bce_loss.item():.4f}"})

        # Early stopping check (based on total loss)
        if current_loss_numpy < best_loss - min_delta:
            best_loss = current_loss_numpy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
            break

    logger.info(f"Training finished. Final loss: {losses[-1]:.4f}")

    # --- Extract Results ---
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        est_difficulties = model.difficulty.cpu().numpy()
        est_log_discriminations = model.log_discrimination.cpu().numpy()
        est_discriminations = np.exp(est_log_discriminations) # Convert back from log scale
        # est_abilities = model.ability.cpu().numpy() # Abilities are also estimated

    # Optional: Center difficulties (like in 1PL) for consistent scale reference
    # est_difficulties = est_difficulties - np.mean(est_difficulties)
    # Centering might interact strangely with discrimination, consider if needed.
    # For now, return the raw estimated difficulty from the 2PL model.

    inv_question_map = {v: k for k, v in question_map.items()}
    results_df = pd.DataFrame({
        'question_idx': range(n_questions),
        'difficulty': est_difficulties,
        'discrimination': est_discriminations
    })
    results_df[question_col] = results_df['question_idx'].map(inv_question_map)

    # Return only question parameters for now
    final_df = results_df[[question_col, 'difficulty', 'discrimination']].sort_values(by=question_col).reset_index(drop=True)

    logger.info("IRT 2PL parameter estimation (PyTorch) completed.")
    return final_df

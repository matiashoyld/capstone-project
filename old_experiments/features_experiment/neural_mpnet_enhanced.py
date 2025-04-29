#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Neural Network Model for Question Difficulty Prediction
This script implements an advanced neural network model that uses:
1. MPNet embeddings for better semantic understanding
2. Rich text features extracted from questions
3. Option similarity metrics
4. Additional categorical and numerical features

The model is designed to predict question difficulty for new questions
where response count data is not available.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
import pickle
import json
import ast  # For safely evaluating the skills string representation
import re
import time
import warnings
from scipy.spatial.distance import cosine, euclidean
from sentence_transformers import SentenceTransformer

# Neural network libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DIFFICULTY_THRESHOLD = -6
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
HIDDEN_DIM = 256
DROPOUT_RATE = 0.3
TFIDF_MAX_FEATURES = 50

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    logger.info("Downloaded NLTK stopwords")

def extract_text_features(text):
    """
    Extract advanced features from text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary of extracted features
    """
    if pd.isna(text) or not isinstance(text, str):
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'digit_count': 0,
            'special_char_count': 0,
            'mathematical_symbols': 0,
            'latex_expressions': 0
        }
    
    # Count words and characters
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    char_count = len(text)
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Count digits
    digit_count = sum(c.isdigit() for c in text)
    
    # Count special characters
    special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
    
    # Count mathematical symbols (including common symbols and operators)
    math_symbols = set(['+', '-', '*', '/', '=', '<', '>', '±', '≤', '≥', '≠', '≈', '∞', '∫', '∑', '∏', '√', '^', '÷', '×', '∆', '∇', '∂'])
    mathematical_symbols = sum(text.count(sym) for sym in math_symbols)
    
    # Count LaTeX expressions (approximate using pattern matching)
    latex_patterns = [r'\\\w+', r'\$.*?\$', r'\\\(.*?\\\)', r'\\\[.*?\\\]']
    latex_expressions = sum(len(re.findall(pattern, text)) for pattern in latex_patterns)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'mathematical_symbols': mathematical_symbols,
        'latex_expressions': latex_expressions
    }

def calculate_option_similarities(row, embedder=None):
    """
    Calculate similarities between question options.
    
    Args:
        row: DataFrame row containing option_a through option_e
        embedder: Sentence embedding model for text similarity
        
    Returns:
        Dictionary of similarity metrics
    """
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    options = [str(row[col]) if pd.notna(row[col]) else "" for col in option_cols]
    valid_options = [opt for opt in options if opt]
    
    # If less than 2 valid options, return zeros
    if len(valid_options) < 2:
        return {
            'avg_text_similarity': 0,
            'max_text_similarity': 0,
            'min_text_similarity': 0,
            'similarity_std': 0
        }
    
    # Calculate text-based similarities
    similarities = []
    
    # If embedder is provided, use semantic similarity
    if embedder:
        try:
            option_embeddings = embedder.encode(valid_options)
            
            # Calculate pairwise similarities
            for i in range(len(valid_options)):
                for j in range(i+1, len(valid_options)):
                    sim = 1 - cosine(option_embeddings[i], option_embeddings[j])
                    similarities.append(sim)
        except:
            # Fall back to length-based similarity if embedding fails
            for i in range(len(valid_options)):
                for j in range(i+1, len(valid_options)):
                    opt1, opt2 = valid_options[i], valid_options[j]
                    len_sim = 1 - abs(len(opt1) - len(opt2)) / max(len(opt1) + len(opt2), 1)
                    similarities.append(len_sim)
    
    # If no embedder or as fallback, use length-based similarity
    else:
        for i in range(len(valid_options)):
            for j in range(i+1, len(valid_options)):
                opt1, opt2 = valid_options[i], valid_options[j]
                len_sim = 1 - abs(len(opt1) - len(opt2)) / max(len(opt1) + len(opt2), 1)
                similarities.append(len_sim)
    
    # Calculate statistics from similarities
    avg_similarity = np.mean(similarities) if similarities else 0
    max_similarity = np.max(similarities) if similarities else 0
    min_similarity = np.min(similarities) if similarities else 0
    std_similarity = np.std(similarities) if len(similarities) > 1 else 0
    
    return {
        'avg_text_similarity': avg_similarity,
        'max_text_similarity': max_similarity,
        'min_text_similarity': min_similarity,
        'similarity_std': std_similarity
    }

class QuestionDataset(Dataset):
    """Dataset for question features and embeddings"""
    
    def __init__(self, features, embeddings, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.embeddings[idx], self.targets[idx]
        else:
            return self.features[idx], self.embeddings[idx]

class EnhancedNeuralDifficultyModel(nn.Module):
    """Enhanced neural network model for predicting question difficulty"""
    
    def __init__(self, features_dim, embedding_dim, hidden_dim=256, dropout_rate=0.3):
        super(EnhancedNeuralDifficultyModel, self).__init__()
        
        # Branch for processing standard features
        self.features_branch = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout_rate)
        )
        
        # Branch for processing embeddings (MPNet has 768 dimensions)
        self.embeddings_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout_rate)
        )
        
        # Combining branches for final prediction with skip connections
        combined_dim = (hidden_dim // 4) * 2  # From both branches
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Residual/skip connection for final layer
        self.residual_connection = nn.Linear(combined_dim, 1)
    
    def forward(self, features, embeddings):
        features_out = self.features_branch(features)
        embeddings_out = self.embeddings_branch(embeddings)
        combined = torch.cat((features_out, embeddings_out), dim=1)
        
        # Main path
        main_out = self.combined_layers(combined)
        
        # Residual connection
        res_out = self.residual_connection(combined)
        
        # Add residual connection
        return (main_out + res_out).squeeze()

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD):
    """
    Load and merge the question data and MPNet embeddings, filtering out very easy questions.
    
    Args:
        difficulty_threshold: Filter out questions with difficulty below this threshold
        
    Returns:
        DataFrame with questions and embeddings
    """
    logger.info("Loading data...")
    
    # Load questions data
    questions_df = pd.read_csv('questions_master.csv')

    # Load IRT difficulties
    irt_difficulties = pd.read_csv('question_difficulties_irt.csv')

    # Merge IRT difficulties into questions dataframe
    questions_df = questions_df.merge(irt_difficulties[['question_id', 'irt_difficulty']], 
                                      on='question_id', 
                                      how='left')
    
    # Check if any questions are missing IRT difficulty values
    missing_irt = questions_df['irt_difficulty'].isna().sum()
    if missing_irt > 0:
        logger.warning(f"{missing_irt} questions are missing IRT difficulty values")
    
    # Filter out extremely easy questions
    total_before = len(questions_df)
    questions_df = questions_df[questions_df['irt_difficulty'] >= difficulty_threshold]
    filtered_count = total_before - len(questions_df)
    logger.info(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
    
    # Load MPNet embeddings
    with open('questions_mpnet_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge dataframes on question_id
    merged_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    logger.info(f"Loaded {len(merged_df)} questions with MPNet embeddings (after filtering)")
    
    return merged_df

def preprocess_features(df):
    """
    Preprocess features with advanced text analysis and similarity metrics.
    
    Args:
        df: DataFrame with questions and embeddings
        
    Returns:
        Processed features, embeddings, target variable, and feature names
    """
    logger.info("Preprocessing features with advanced engineering...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Extract MPNet embeddings
    embeddings = np.array(df_processed['embedding'].tolist())
    logger.info(f"MPNet embedding dimension: {embeddings.shape[1]}")
    
    # Process skills
    logger.info("Processing skills...")
    mlb = MultiLabelBinarizer()
    skills_lists = df_processed['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    skills_binary = mlb.fit_transform(skills_lists)
    skills_df = pd.DataFrame(skills_binary, columns=mlb.classes_)
    
    # Convert all column names to strings
    skills_df.columns = skills_df.columns.astype(str)
    
    # Basic features without ID features (as requested)
    logger.info("Extracting basic features...")
    basic_features = ['level', 'num_misconceptions', 'has_image', 'avg_steps']
    
    # Note: Excluding topic_id, subject_id, axis_id as requested
    
    # Extract text features from question title
    logger.info("Extracting text features...")
    text_features = df_processed['question_title'].apply(
        lambda x: extract_text_features(x) if pd.notna(x) else extract_text_features("")
    )
    
    # Convert text features to DataFrame
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Create interaction features
    logger.info("Creating interaction features...")
    df_processed['math_to_text_ratio'] = text_features_df['mathematical_symbols'] / (text_features_df['word_count'] + 1)
    df_processed['digit_to_text_ratio'] = text_features_df['digit_count'] / (text_features_df['word_count'] + 1)
    df_processed['special_to_text_ratio'] = text_features_df['special_char_count'] / (text_features_df['word_count'] + 1)
    df_processed['latex_to_text_ratio'] = text_features_df['latex_expressions'] / (text_features_df['word_count'] + 1)
    
    # Question text length and complexity
    df_processed['question_length'] = df_processed['question_title'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Option complexity (average length of options)
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    df_processed['avg_option_length'] = df_processed[option_cols].apply(
        lambda row: np.mean([len(str(x)) for x in row if pd.notna(x)]), axis=1
    )
    
    # Number of options
    df_processed['num_options'] = df_processed[option_cols].apply(
        lambda row: sum(1 for x in row if pd.notna(x)), axis=1
    )
    
    # Calculate similarity between options
    logger.info("Calculating option similarities...")
    # Initialize a lightweight sentence embedder for similarity calculations
    option_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Calculate option similarities in batches to improve performance
    similarities = []
    batch_size = 500
    for i in range(0, len(df_processed), batch_size):
        batch = df_processed.iloc[i:i+batch_size]
        batch_similarities = batch.apply(lambda row: calculate_option_similarities(row, option_embedder), axis=1)
        similarities.extend(batch_similarities)
    
    # Convert similarities to DataFrame
    similarities_df = pd.DataFrame(similarities)
    
    # Generate TF-IDF features from question text
    logger.info("Generating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, 
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2)
    )
    
    # Fill NA values in question_title with empty string
    question_texts = df_processed['question_title'].fillna("").astype(str)
    tfidf_matrix = tfidf.fit_transform(question_texts)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    )
    
    # List of all features to include
    text_feature_cols = [
        'word_count', 'char_count', 'avg_word_length',
        'digit_count', 'special_char_count', 'mathematical_symbols',
        'latex_expressions', 'math_to_text_ratio', 'digit_to_text_ratio',
        'special_to_text_ratio', 'latex_to_text_ratio',
        'question_length', 'avg_option_length', 'num_options'
    ]
    
    # Combine all text features into processed dataframe
    for col in text_features_df.columns:
        df_processed[col] = text_features_df[col]
    
    # Combine features
    logger.info("Combining all features...")
    feature_dfs = [
        df_processed[basic_features + text_feature_cols].reset_index(drop=True),
        skills_df.reset_index(drop=True),
        similarities_df.reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ]
    
    X_without_embeddings = pd.concat(feature_dfs, axis=1)
    
    # Scale features
    logger.info("Scaling features...")
    feature_names = X_without_embeddings.columns.tolist()
    
    # Use Yeo-Johnson power transformation for better normalization
    scaler = PowerTransformer(method='yeo-johnson')
    X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings.fillna(0))
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
        'scaler': scaler,
        'tfidf': tfidf
    }
    
    with open('feature_processors_neural_mpnet_enhanced.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    logger.info(f"Processed features shape: {X_without_embeddings_scaled.shape}")
    logger.info(f"Number of feature names: {len(feature_names)}")
    
    return X_without_embeddings_scaled, embeddings, y, feature_names

def train_model(X, y, embeddings, feature_names=None, batch_size=BATCH_SIZE, 
               epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=PATIENCE, 
               weight_decay=1e-5, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE, 
               cv_folds=CV_FOLDS):
    """
    Train and evaluate the neural network model using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        embeddings: Embedding vectors
        feature_names: Names of features (for reporting)
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Patience for early stopping
        weight_decay: L2 regularization strength
        hidden_dim: Size of hidden layers
        dropout_rate: Dropout rate
        cv_folds: Number of cross-validation folds
        
    Returns:
        best_model: Best trained model
        cv_rmse_scores: Cross-validation RMSE scores
        test_rmse: RMSE on test set
        test_mae: MAE on test set
        test_r2: R² on test set
    """
    logger.info(f"Training model with {X.shape[1]} features and {embeddings.shape[1]}-dimensional embeddings")
    
    # Split the data into training and testing sets
    X_train, X_test, embeddings_train, embeddings_test, y_train, y_test = train_test_split(
        X, embeddings, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # For cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_rmse_scores = []
    
    # For storing the best model
    best_val_rmse = float('inf')
    best_model = None
    
    # Cross-validation
    fold = 1
    for train_idx, val_idx in kf.split(X_train):
        logger.info(f"Training fold {fold}/{cv_folds}")
        fold_start_time = time.time()
        
        # Split training data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        embeddings_fold_train, embeddings_fold_val = embeddings_train[train_idx], embeddings_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create datasets
        train_dataset = QuestionDataset(X_fold_train, embeddings_fold_train, y_fold_train)
        val_dataset = QuestionDataset(X_fold_val, embeddings_fold_val, y_fold_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = EnhancedNeuralDifficultyModel(
            features_dim=X.shape[1],
            embedding_dim=embeddings.shape[1],
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        best_fold_val_rmse = float('inf')
        best_fold_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_losses = []
            
            for features, embedding, target in train_loader:
                features, embedding, target = features.to(device), embedding.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(features, embedding)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for features, embedding, target in val_loader:
                    features, embedding, target = features.to(device), embedding.to(device), target.to(device)
                    output = model(features, embedding)
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
            
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            avg_train_loss = np.mean(train_losses)
            
            # Update learning rate
            scheduler.step(val_rmse)
            
            # Check for improvement
            if val_rmse < best_fold_val_rmse:
                best_fold_val_rmse = val_rmse
                best_fold_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= patience:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val RMSE = {val_rmse:.4f}, Best Val RMSE = {best_fold_val_rmse:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best epoch was {best_fold_epoch} with Val RMSE = {best_fold_val_rmse:.4f}")
                break
        
        cv_rmse_scores.append(best_fold_val_rmse)
        
        # Save the best model across all folds
        if best_fold_val_rmse < best_val_rmse:
            best_val_rmse = best_fold_val_rmse
            
            # Create a fresh model with the same architecture for final evaluation
            best_model = EnhancedNeuralDifficultyModel(
                features_dim=X.shape[1],
                embedding_dim=embeddings.shape[1],
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            ).to(device)
            
            # Copy the trained parameters
            best_model.load_state_dict(model.state_dict())
        
        fold_time = time.time() - fold_start_time
        logger.info(f"Fold {fold} completed in {fold_time:.2f} seconds with best RMSE = {best_fold_val_rmse:.4f}")
        fold += 1
    
    # Calculate cross-validation statistics
    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    logger.info(f"Cross-validation RMSE: {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    
    # Evaluate on test set
    test_dataset = QuestionDataset(X_test, embeddings_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    best_model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features, embedding, target in test_loader:
            features, embedding, target = features.to(device), embedding.to(device), target.to(device)
            output = best_model(features, embedding)
            test_preds.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('MPNet Enhanced Neural Network: Actual vs Predicted Difficulty')
    plt.tight_layout()
    plt.savefig('mpnet_neural_actual_vs_predicted.png')
    logger.info("Actual vs Predicted plot saved to 'mpnet_neural_actual_vs_predicted.png'")
    
    # Save the best model
    torch.save(best_model.state_dict(), 'mpnet_neural_model.pt')
    logger.info("Model saved to 'mpnet_neural_model.pt'")
    
    return best_model, cv_rmse_scores, test_rmse, test_mae, test_r2

def update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with the results from this run.
    
    Args:
        cv_scores: Cross-validation scores
        test_rmse: RMSE on test set
        test_mae: MAE on test set
        test_r2: R² on test set
    """
    mean_cv_rmse = np.mean(cv_scores)
    std_cv_rmse = np.std(cv_scores)
    
    metrics = {
        "MPNet Enhanced Neural Network": {
            "Mean CV RMSE": f"{mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}",
            "Test RMSE": f"{test_rmse:.4f}",
            "Test MAE": f"{test_mae:.4f}",
            "Test R²": f"{test_r2:.4f}"
        }
    }
    
    # Create or update comparison file
    comparison_file = 'model_metrics_comparison.json'
    
    try:
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        comparison = {}
    
    # Update with new metrics
    comparison.update(metrics)
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Metrics comparison updated in '{comparison_file}'")

def main():
    """Main function to run the enhanced neural model pipeline."""
    start_time = time.time()
    
    # Load data
    df = load_data()
    
    # Preprocess features
    X, embeddings, y, feature_names = preprocess_features(df)
    
    # Train and evaluate model
    model, cv_scores, test_rmse, test_mae, test_r2 = train_model(
        X, y, embeddings, feature_names,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        cv_folds=CV_FOLDS
    )
    
    # Update metrics comparison
    update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2)
    
    # Print total runtime
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")
    
    # Print summary of results for easy comparison
    logger.info("\n===== MPNet Enhanced Neural Network Results =====")
    logger.info(f"Cross-validation RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info("============================================")

if __name__ == "__main__":
    main() 
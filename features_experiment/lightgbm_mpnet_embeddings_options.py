#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM Model for Question Difficulty Prediction (With MPNet Embeddings and Option Embeddings)
This script implements a LightGBM model that uses:
1. Basic text features extracted from questions
2. Cosine similarity between correct option and wrong options
3. Basic categorical and numerical features
4. MPNet embeddings for semantic understanding

The model is designed to predict question difficulty for new questions
incorporating embeddings for better semantic understanding.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import lightgbm as lgb
import re
import time
import warnings
import os
import pickle
from datetime import datetime
import optuna  # Import for hyperparameter optimization
from functools import partial  # For passing arguments to objective function
import json  # Import for saving best parameters
from scipy.spatial.distance import cosine  # For cosine similarity

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DIFFICULTY_THRESHOLD = -6
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5
# LightGBM specific parameters
NUM_ITERATIONS = 1000
EARLY_STOPPING_ROUNDS = 50
# Hyperparameter optimization
N_TRIALS = 20  # Number of hyperparameter optimization trials

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

def extract_text_features(text):
    """
    Extract basic text features from a text string.
    
    Args:
        text: Text string to analyze
        
    Returns:
        Dictionary of text features
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
    
    # Basic counts
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    char_count = len(text)
    
    # Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0
    
    # Count digits
    digit_count = sum(c.isdigit() for c in text)
    
    # Count special characters
    special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
    
    # Count mathematical symbols
    mathematical_symbols = len(re.findall(r'[\+\-\*\/\=\<\>\^\(\)\[\]\{\}\%]', text))
    
    # Count LaTeX expressions
    latex_expressions = len(re.findall(r'\$.*?\$', text))
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'mathematical_symbols': mathematical_symbols,
        'latex_expressions': latex_expressions
    }

def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score (1 - cosine distance)
    """
    if v1 is None or v2 is None:
        return 0
    
    # Compute cosine similarity (1 - cosine distance)
    return 1 - cosine(v1, v2)

def calculate_option_similarities(row):
    """
    Calculate cosine similarities between correct option and wrong options.
    
    Args:
        row: DataFrame row containing option embeddings and correct_option_letter
        
    Returns:
        Dictionary of similarity metrics
    """
    # Get the correct option letter
    correct_letter = row['correct_option_letter'].upper() if pd.notna(row['correct_option_letter']) else None
    
    if correct_letter is None:
        return {
            'cosine_sim_correct_wrong1': 0,
            'cosine_sim_correct_wrong2': 0,
            'cosine_sim_correct_wrong3': 0,
            'cosine_sim_correct_wrong4': 0
        }
    
    # Get the correct option embedding
    correct_emb_col = f'option_{correct_letter.lower()}_embedding'
    correct_emb = row[correct_emb_col]
    
    if correct_emb is None:
        return {
            'cosine_sim_correct_wrong1': 0,
            'cosine_sim_correct_wrong2': 0,
            'cosine_sim_correct_wrong3': 0,
            'cosine_sim_correct_wrong4': 0
        }
    
    # Get all option letters
    all_letters = ['a', 'b', 'c', 'd', 'e']
    wrong_letters = [letter for letter in all_letters if letter != correct_letter.lower()]
    
    # Calculate similarities between correct and each wrong option
    similarities = {}
    for i, letter in enumerate(wrong_letters, 1):
        wrong_emb_col = f'option_{letter}_embedding'
        wrong_emb = row[wrong_emb_col]
        
        # Calculate similarity if the wrong option exists
        if wrong_emb is not None:
            sim = cosine_similarity(correct_emb, wrong_emb)
        else:
            sim = 0
        
        # Store the similarity with a fixed key regardless of which options are used
        similarities[f'cosine_sim_correct_wrong{i}'] = sim
    
    # Ensure we have all 4 similarity features (padded with zeros if necessary)
    for i in range(1, 5):
        if f'cosine_sim_correct_wrong{i}' not in similarities:
            similarities[f'cosine_sim_correct_wrong{i}'] = 0
    
    return similarities

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD):
    """
    Load the question data and MPNet embeddings, filtering out very easy questions.
    
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
    
    # Load MPNet embeddings with option embeddings
    with open('questions_mpnet_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge dataframes on question_id
    merged_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    logger.info(f"Loaded {len(merged_df)} questions with MPNet embeddings (after filtering)")
    
    return merged_df

def preprocess_features(df):
    """
    Preprocess features with text analysis, similarity metrics, and embeddings.
    
    Args:
        df: DataFrame with questions and MPNet embeddings
        
    Returns:
        Tuple of (X, y, feature_names) where X is the feature matrix and y is the target variable
    """
    logger.info("Preprocessing features...")
    df_processed = df.copy()
    
    # Extract embeddings from DataFrame
    formatted_embeddings = np.array(df_processed['formatted_embedding'].tolist())
    
    # Basic features
    basic_features = ['subject_id', 'level', 'num_misconceptions', 'has_image', 'avg_steps']
    categorical_features = ['subject_id']
    
    # Extract text features from question titles
    logger.info("Extracting text features...")
    text_features = df_processed['question_title'].apply(extract_text_features)
    
    # Convert text features to DataFrame
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Question text length
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
    
    # Calculate cosine similarity between correct and wrong options
    logger.info("Calculating option similarities...")
    similarities = df_processed.apply(calculate_option_similarities, axis=1)
    similarities_df = pd.DataFrame(similarities.tolist())
    
    # List of all features to include
    text_feature_cols = [
        'word_count', 'char_count', 'avg_word_length',
        'digit_count', 'special_char_count', 'mathematical_symbols',
        'latex_expressions', 'question_length', 'avg_option_length', 'num_options'
    ]
    
    # Combine all text features into processed dataframe
    for col in text_features_df.columns:
        df_processed[col] = text_features_df[col]
    
    # Combine features without embeddings
    logger.info("Combining features...")
    feature_dfs = [
        df_processed[basic_features + text_feature_cols].reset_index(drop=True),
        similarities_df.reset_index(drop=True)
    ]
    
    X_without_embeddings = pd.concat(feature_dfs, axis=1)
    
    # Fill NA values
    X_without_embeddings = X_without_embeddings.fillna(0)
    
    # Scale features
    logger.info("Scaling features...")
    feature_names = X_without_embeddings.columns.tolist()
    
    # Use Yeo-Johnson power transformation for better normalization
    scaler = PowerTransformer(method='yeo-johnson')
    X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings)
    
    # Apply PCA to reduce dimensionality of embeddings
    logger.info("Applying PCA to reduce embedding dimensions...")
    pca = PCA(n_components=50)
    embeddings_reduced = pca.fit_transform(formatted_embeddings)
    
    # Create final feature matrix including embeddings
    X_with_embeddings = np.hstack([X_without_embeddings_scaled, embeddings_reduced])
    
    # Update feature names to include embedding dimensions
    embedding_feature_names = [f'embedding_pca_{i}' for i in range(embeddings_reduced.shape[1])]
    all_feature_names = feature_names + embedding_feature_names
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'scaler': scaler,
        'pca': pca
    }
    
    with open('feature_processors_lightgbm_mpnet_embeddings_options.pkl', 'wb') as f:
        pickle.dump(processors, f)
        
    return X_with_embeddings, y, all_feature_names

def create_difficulty_bins(y, n_bins=10):
    """
    Create stratified bins for the difficulty values to ensure
    balanced representation in the folds during cross-validation.
    
    Args:
        y: Target difficulty values
        n_bins: Number of bins to create
        
    Returns:
        Bin assignments for each sample
    """
    # Create equal-sized bins based on percentiles
    bins = np.percentile(y, np.linspace(0, 100, n_bins+1))
    bin_assignments = np.digitize(y, bins[1:-1])
    return bin_assignments

def objective(trial, X, y, feature_names):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        
    Returns:
        Mean RMSE across folds
    """
    # Create bins for stratification
    bins = create_difficulty_bins(y)
    
    # Define hyperparameters to optimize
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'verbose': -1
    }
    
    # Use stratified k-fold cross-validation
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X, bins):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        ]
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=callbacks,
            num_boost_round=NUM_ITERATIONS
        )
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
    
    # Return mean RMSE
    return np.mean(rmse_scores)

def hyperparameter_optimization(X, y, feature_names):
    """
    Perform hyperparameter optimization using Optuna.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        
    Returns:
        Dictionary of best hyperparameters
    """
    logger.info(f"Starting hyperparameter optimization with {N_TRIALS} trials...")
    
    # Create a partial function with fixed arguments
    objective_func = partial(objective, X=X, y=y, feature_names=feature_names)
    
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    
    # Add callback to print progress
    def print_progress(study, trial):
        """Callback to print trial progress."""
        if trial.number % 1 == 0:  # Print every trial
            logger.info(f"Trial {trial.number}: Current RMSE = {trial.value:.4f}, " +
                       f"Best RMSE = {study.best_value:.4f}")
    
    study.optimize(objective_func, n_trials=N_TRIALS, callbacks=[print_progress])
    
    # Get best parameters
    best_params = study.best_params.copy()
    
    # Add fixed parameters
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['verbose'] = -1
    
    # Log best parameters and score
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    
    # Save best parameters to file
    best_params_file = "mpnet_embeddings_best_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Best parameters saved to {best_params_file}")
    
    return best_params

def train_with_best_params(X, y, feature_names, best_params):
    """
    Train and evaluate a model with the best hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        best_params: Dictionary of best hyperparameters
        
    Returns:
        Trained model, RMSE, MAE, and R² on test set
    """
    logger.info("Training model with best hyperparameters...")
    
    # Split data into train and test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=100, show_stdv=False)
    ]
    
    model = lgb.train(
        best_params,
        train_data,
        valid_sets=[test_data],
        callbacks=callbacks,
        num_boost_round=NUM_ITERATIONS
    )
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics on test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test set with best hyperparameters - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Log feature importance
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Log top 10 features
    logger.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Save feature importance to CSV
    feature_importance_file = "feature_importance_mpnet_embeddings_optimized.csv"
    feature_importance_df.to_csv(feature_importance_file, index=False)
    logger.info(f"Feature importance saved to {feature_importance_file}")
    
    # Create scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Optimized LightGBM with MPNet Embeddings: Actual vs Predicted Difficulty')
    plt.savefig('lightgbm_mpnet_embeddings_optimized_predictions.png')
    plt.close()
    
    # Save model
    model_file = "lightgbm_mpnet_embeddings_optimized.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_file}")
    
    # Update metrics comparison
    update_metrics_comparison("LightGBM_MPNet_Embeddings_Optimized", test_rmse, test_mae, test_r2)
    
    return model, test_rmse, test_mae, test_r2

def train_and_evaluate_model(X, y, feature_names):
    """
    Train the LightGBM model with hyperparameter optimization and evaluate performance.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Names of features
        
    Returns:
        Tuple of (RMSE, MAE, R²) on test set
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Hyperparameter optimization
    logger.info("Starting hyperparameter optimization...")
    best_params = hyperparameter_optimization(X_train, y_train, feature_names)
    
    # Save best parameters to JSON
    with open('best_params_lightgbm_mpnet_embeddings_options.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Train final model with best hyperparameters
    logger.info("Training final model with best hyperparameters...")
    model = train_with_best_params(X_train, y_train, feature_names, best_params)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test set with best hyperparameters - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Log feature importance
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Log top 10 features
    logger.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Save feature importance to CSV
    feature_importance_file = "feature_importance_mpnet_embeddings_options.csv"
    feature_importance_df.to_csv(feature_importance_file, index=False)
    logger.info(f"Feature importance saved to {feature_importance_file}")
    
    # Create scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Optimized LightGBM with MPNet Embeddings and Option Cosine Similarities: Actual vs Predicted Difficulty')
    plt.savefig('lightgbm_mpnet_embeddings_options_predictions.png')
    plt.close()
    
    # Save model
    model_file = "lightgbm_mpnet_embeddings_options.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_file}")
    
    # Update metrics comparison
    update_metrics_comparison("MPNet Embeddings with Option Cosine Similarities", test_rmse, test_mae, test_r2)
    
    return test_rmse, test_mae, test_r2

def update_metrics_comparison(model_name, rmse, mae, r2):
    """
    Update the metrics comparison CSV file with results from a new model.
    
    Args:
        model_name: Name of the model
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared
    """
    metrics_file = "model_metrics_comparison.csv"
    
    # Create DataFrame for current model
    current_metrics = pd.DataFrame({
        'Model': [model_name],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2],
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    
    # Check if metrics file exists
    if os.path.exists(metrics_file):
        # Read existing metrics
        metrics_df = pd.read_csv(metrics_file)
        
        # Append new metrics
        metrics_df = pd.concat([metrics_df, current_metrics], ignore_index=True)
    else:
        # Create new metrics DataFrame
        metrics_df = current_metrics
    
    # Save updated metrics
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Updated model metrics comparison in {metrics_file}")
    
    # Create comparison plot of RMSE
    plt.figure(figsize=(12, 6))
    metrics_df = metrics_df.sort_values('RMSE')
    plt.barh(metrics_df['Model'], metrics_df['RMSE'])
    plt.xlabel('RMSE (lower is better)')
    plt.title('Model RMSE Comparison')
    plt.tight_layout()
    plt.savefig('model_rmse_comparison.png')
    plt.close()

def main():
    """
    Main function that orchestrates the entire pipeline.
    
    1. Load and preprocess data
    2. Perform hyperparameter optimization
    3. Train final model with best hyperparameters
    4. Save results and visualizations
    """
    # Start timer
    start_time = time.time()
    
    # Load data
    df = load_data()
    
    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Preprocess features
    X, y, feature_names = preprocess_features(df)
    
    # Train and evaluate model
    test_rmse, test_mae, test_r2 = train_and_evaluate_model(X, y, feature_names)
    
    # Update metrics comparison
    update_metrics_comparison("MPNet Embeddings with Option Cosine Similarities", test_rmse, test_mae, test_r2)
    
    # End timer
    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 
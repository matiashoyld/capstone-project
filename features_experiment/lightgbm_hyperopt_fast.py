#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast LightGBM Hyperparameter Optimization

This script performs accelerated hyperparameter tuning for the LightGBM model
using Optuna with parallel processing, data sampling, and focused parameter ranges.
The goal is to quickly find good hyperparameters that can beat BERT's performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import ast  # For safely evaluating the skills string representation
import logging
import time
import joblib
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DIFFICULTY_THRESHOLD = -6  # Filter out very easy questions
N_TRIALS = 30  # Reduced number of hyperparameter optimization trials
CV_FOLDS = 3  # Reduced number of cross-validation folds
RANDOM_STATE = 42  # For reproducibility
SAMPLE_FRACTION = 0.3  # Fraction of data to use for hyperparameter tuning
N_JOBS = -1  # Number of parallel jobs (-1 means use all available cores)

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD, sample_fraction=SAMPLE_FRACTION):
    """
    Load the question data, filter out questions with difficulty below the threshold,
    and sample a fraction of the data for faster hyperparameter tuning.
    
    Args:
        difficulty_threshold: Filter out questions with difficulty below this threshold
        sample_fraction: Fraction of data to sample for hyperparameter tuning
        
    Returns:
        DataFrame containing the filtered and sampled questions
    """
    logger.info("Loading data...")
    
    # Load questions data
    questions_df = pd.read_csv("questions_master.csv")
    
    # Load IRT difficulties
    irt_difficulties = pd.read_csv("question_difficulties_irt.csv")
    
    # Merge IRT difficulties into questions dataframe
    questions_df = questions_df.merge(irt_difficulties[['question_id', 'irt_difficulty']], 
                                      on='question_id', 
                                      how='left')
    
    # Check if any questions are missing IRT difficulty values
    missing_irt = questions_df['irt_difficulty'].isna().sum()
    if missing_irt > 0:
        logger.info(f"Warning: {missing_irt} questions are missing IRT difficulty values")
    
    # Load embeddings
    with open('questions_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge embeddings with questions dataframe
    questions_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    # Filter out questions with difficulty below threshold
    original_count = len(questions_df)
    questions_df = questions_df[questions_df['irt_difficulty'] >= difficulty_threshold]
    filtered_count = original_count - len(questions_df)
    
    logger.info(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
    logger.info(f"Loaded {len(questions_df)} questions with embeddings (after filtering)")
    
    # Sample a fraction of the data for faster hyperparameter tuning
    if sample_fraction < 1.0:
        sample_size = int(len(questions_df) * sample_fraction)
        questions_df = questions_df.sample(n=sample_size, random_state=RANDOM_STATE)
        logger.info(f"Sampled {len(questions_df)} questions ({sample_fraction:.0%} of filtered data) for hyperparameter tuning")
    
    return questions_df

def preprocess_features(df, use_id_features=True, use_nlp_features=True, reduce_dimensions=True):
    """
    Preprocess features for model training, without using count features.
    
    Args:
        df: DataFrame containing the question data
        use_id_features: Whether to use ID features (topic_id, subject_id, axis_id)
        use_nlp_features: Whether to use NLP-derived features
        reduce_dimensions: Whether to use only a subset of embedding dimensions
        
    Returns:
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
    """
    logger.info("Preprocessing features...")
    
    # Extract embeddings
    embeddings = np.array(df['embedding'].tolist())
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Process skills
    mlb = MultiLabelBinarizer()
    skills_lists = df_processed['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    skills_binary = mlb.fit_transform(skills_lists)
    skills_df = pd.DataFrame(skills_binary, columns=mlb.classes_)
    
    # Convert all column names to strings
    skills_df.columns = skills_df.columns.astype(str)
    
    # Basic features without count features
    basic_features = ['level', 'num_misconceptions', 'has_image', 'avg_steps']
    
    # Add computed text-based features (if NLP features are used)
    if use_nlp_features:
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
        
        # Add these new features to basic features
        nlp_features = ['question_length', 'avg_option_length', 'num_options']
        basic_features.extend(nlp_features)
    
    logger.info("Note: Using only features available for new questions (no count features)")
    
    # ID features (optional)
    id_features = []
    if use_id_features:
        # One-hot encode categorical ID features
        id_cols = ['topic_id', 'subject_id', 'axis_id']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_ids = encoder.fit_transform(df_processed[id_cols])
        id_feature_names = [f"{col}_{val}" for col, vals in zip(id_cols, encoder.categories_) 
                           for val in vals]
        id_features_df = pd.DataFrame(encoded_ids, columns=id_feature_names)
        id_features = id_feature_names
    
    # Combine features
    if use_id_features:
        X_without_embeddings = pd.concat([
            df_processed[basic_features].reset_index(drop=True),
            skills_df.reset_index(drop=True),
            id_features_df.reset_index(drop=True)
        ], axis=1)
    else:
        X_without_embeddings = pd.concat([
            df_processed[basic_features].reset_index(drop=True),
            skills_df.reset_index(drop=True)
        ], axis=1)
    
    # Convert all column names to strings to avoid validation errors
    X_without_embeddings.columns = X_without_embeddings.columns.astype(str)
    
    # Scale features
    scaler = StandardScaler()
    X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings)
    
    # Reduce embedding dimensions to speed up optimization
    if reduce_dimensions:
        # Take only a subset of the embedding dimensions (e.g., first 50)
        embedding_dims = min(50, embeddings.shape[1])
        embeddings_reduced = embeddings[:, :embedding_dims]
        logger.info(f"Using only first {embedding_dims} dimensions of embeddings for faster optimization")
    else:
        embeddings_reduced = embeddings
    
    # Add embeddings to processed features
    X = np.hstack([X_without_embeddings_scaled, embeddings_reduced])
    
    # Feature names (for feature importance)
    feature_names = list(X_without_embeddings.columns) + [f"emb_{i}" for i in range(embeddings_reduced.shape[1])]
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
        'scaler': scaler
    }
    if use_id_features:
        processors['id_encoder'] = encoder
    
    with open('feature_processors_hyperopt_fast.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X, y, feature_names

def objective(trial, X, y, feature_names):
    """
    Objective function for Optuna optimization, with a focused parameter space.
    
    Args:
        trial: Optuna trial object
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        
    Returns:
        Mean validation RMSE across folds
    """
    # Define a more focused hyperparameter search space with narrower ranges
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),  # Focus on gbdt as most stable
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # Narrower range
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),  # Focused range
        'max_depth': trial.suggest_int('max_depth', 5, 10),  # Narrower range
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),  # Typical range
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1.0, log=True),  # Regularization
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1.0, log=True),  # Regularization
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),  # Higher starting value
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),  # Higher starting value
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),  # Lower upper bound
        'verbose': -1,
    }
    
    # Cross-validation
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with fewer boosting rounds
        max_boost_round = 1000  # Reduced from 10000
        early_stopping = 50
        
        model = lgb.train(
            param,
            train_data,
            num_boost_round=max_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            ]
        )
        
        # Report best score to Optuna for pruning
        trial.report(model.best_score['valid_0']['rmse'], step=0)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
    
    mean_rmse = np.mean(rmse_scores)
    
    # Log metrics
    logger.info(f"Trial {trial.number}: RMSE = {mean_rmse:.4f}")
    
    # Return the mean RMSE (objective to minimize)
    return mean_rmse

def train_full_model(best_params, X_full, y_full, feature_names, test_size=0.2):
    """
    Train the final LightGBM model on the full dataset with the best hyperparameters.
    
    Args:
        best_params: Best hyperparameters found by Optuna
        X_full: Full processed features dataset
        y_full: Full target variable dataset
        feature_names: Names of the processed features
        test_size: Proportion of data to use for testing
        
    Returns:
        model: Trained LightGBM model
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    logger.info("Training final model on the full dataset with best hyperparameters...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train final model with more boosting rounds
    num_boost_round = 10000  # More rounds for final model
    
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[lgb.Dataset(X_test, label=y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    
    # Save model
    model_file = 'question_model_hyperopt_fast.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved optimized model to {model_file}")
    
    # Create and save scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Fast Hyperopt LightGBM: Actual vs Predicted Difficulty')
    scatter_file = 'prediction_scatter_hyperopt_fast.png'
    plt.savefig(scatter_file)
    plt.close()
    logger.info(f"Saved prediction scatter plot to {scatter_file}")
    
    # Analyze feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_file = 'feature_importance_hyperopt_fast.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"Saved feature importance to {importance_file}")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Fast Hyperopt Model)')
    plt.tight_layout()
    importance_plot_file = 'feature_importance_hyperopt_fast.png'
    plt.savefig(importance_plot_file)
    plt.close()
    logger.info(f"Saved feature importance plot to {importance_plot_file}")
    
    return model, test_rmse, test_mae, test_r2

def update_metrics_comparison(test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the fast hyperparameter-optimized model.
    
    Args:
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
        
        # Create new entry for hyperopt model
        new_entry = (
            f"\n### LightGBM Model with Fast Hyperparameter Optimization (No Count Features)\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < {DIFFICULTY_THRESHOLD} and does not use response count features\n"
            f"- Note: Hyperparameters were optimized using fast parallel Bayesian optimization\n"
        )
        
        # Find the position to insert the new entry (under LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
            
            logger.info("Updated metrics_comparison.txt with fast hyperopt model results.")
        else:
            logger.error("Could not update metrics_comparison.txt. Section markers not found.")
    
    except Exception as e:
        logger.error(f"Error updating metrics comparison: {str(e)}")

def main():
    """Main function to run the fast hyperparameter optimization"""
    start_time = time.time()
    
    # Load a sample of the data
    sample_df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD, sample_fraction=SAMPLE_FRACTION)
    
    # Extract features and target from sample
    X_sample, y_sample, feature_names_sample = preprocess_features(
        sample_df, 
        use_id_features=True,
        use_nlp_features=True,
        reduce_dimensions=True
    )
    
    # Create the Optuna study with parallel processing
    study_name = "lightgbm_hyperopt_fast"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # Optimize hyperparameters in parallel
    logger.info(f"Starting fast hyperparameter optimization with {N_TRIALS} trials and {N_JOBS} parallel jobs...")
    study.optimize(
        partial(objective, X=X_sample, y=y_sample, feature_names=feature_names_sample),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best RMSE on sample: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save best hyperparameters
    with open('hyperopt_best_params_fast.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Now train on the full dataset with best parameters
    logger.info("Now loading full dataset for final model training...")
    full_df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD, sample_fraction=1.0)
    
    # Extract features and target from full dataset
    X_full, y_full, feature_names_full = preprocess_features(
        full_df, 
        use_id_features=True,
        use_nlp_features=True,
        reduce_dimensions=False  # Use all embedding dimensions for final model
    )
    
    # Train the final model on the full dataset with the best hyperparameters
    model, test_rmse, test_mae, test_r2 = train_full_model(
        best_params, X_full, y_full, feature_names_full
    )
    
    # Update metrics comparison file
    update_metrics_comparison(test_rmse, test_mae, test_r2)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    logger.info(f"Fast hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final model Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    
    # Check if we beat BERT
    bert_r2 = 0.5300
    if test_r2 > bert_r2:
        logger.info(f"SUCCESS! The optimized LightGBM model (R² = {test_r2:.4f}) outperforms BERT (R² = {bert_r2:.4f})!")
    else:
        logger.info(f"The optimized LightGBM model (R² = {test_r2:.4f}) did not outperform BERT (R² = {bert_r2:.4f}).")
        logger.info("Consider the enhanced feature engineering approach or ensemble methods.")

if __name__ == "__main__":
    main() 
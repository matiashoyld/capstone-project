#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM Hyperparameter Optimization

This script performs advanced hyperparameter tuning for the LightGBM model
using Optuna for Bayesian optimization. The goal is to optimize the model
without using response count features to beat BERT's performance.
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
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DIFFICULTY_THRESHOLD = -6  # Filter out very easy questions
N_TRIALS = 100  # Number of hyperparameter optimization trials
CV_FOLDS = 5  # Number of cross-validation folds
RANDOM_STATE = 42  # For reproducibility

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD):
    """
    Load the question data and filter out questions with difficulty below the threshold.
    
    Args:
        difficulty_threshold: Filter out questions with difficulty below this threshold
        
    Returns:
        DataFrame containing the filtered questions
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
    
    return questions_df

def preprocess_features(df, use_id_features=True, use_nlp_features=True):
    """
    Preprocess features for model training, without using count features.
    
    Args:
        df: DataFrame containing the question data
        use_id_features: Whether to use ID features (topic_id, subject_id, axis_id)
        use_nlp_features: Whether to use NLP-derived features
        
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
    
    # Add embeddings to processed features
    X = np.hstack([X_without_embeddings_scaled, embeddings])
    
    # Feature names (for feature importance)
    feature_names = list(X_without_embeddings.columns) + [f"emb_{i}" for i in range(embeddings.shape[1])]
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
        'scaler': scaler
    }
    if use_id_features:
        processors['id_encoder'] = encoder
    
    with open('feature_processors_hyperopt.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X, y, feature_names

def objective(trial, X, y, feature_names):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        
    Returns:
        Mean validation RMSE across folds
    """
    # Define the hyperparameter search space
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        'verbose': -1,
    }
    
    # If boosting_type is 'goss', bagging-related parameters are not used
    if param['boosting_type'] == 'goss':
        param.pop('bagging_fraction')
        param.pop('bagging_freq')
    
    # Cross-validation
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        num_boost_round = 10000
        model = lgb.train(
            param,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
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
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    
    # Log metrics
    logger.info(f"Trial {trial.number}: RMSE = {mean_rmse:.4f}, R² = {mean_r2:.4f}")
    
    # Return the mean RMSE (objective to minimize)
    return mean_rmse

def train_final_model(best_params, X, y, feature_names, test_size=0.2):
    """
    Train the final LightGBM model with the best hyperparameters.
    
    Args:
        best_params: Best hyperparameters found by Optuna
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        test_size: Proportion of data to use for testing
        
    Returns:
        model: Trained LightGBM model
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    logger.info("Training final model with best hyperparameters...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train final model
    num_boost_round = 10000
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=num_boost_round
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
    model_file = 'question_model_hyperopt.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved optimized model to {model_file}")
    
    # Create and save scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Hyperopt LightGBM: Actual vs Predicted Difficulty')
    scatter_file = 'prediction_scatter_hyperopt.png'
    plt.savefig(scatter_file)
    plt.close()
    logger.info(f"Saved prediction scatter plot to {scatter_file}")
    
    # Analyze feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_file = 'feature_importance_hyperopt.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"Saved feature importance to {importance_file}")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Hyperopt Model)')
    plt.tight_layout()
    importance_plot_file = 'feature_importance_hyperopt.png'
    plt.savefig(importance_plot_file)
    plt.close()
    logger.info(f"Saved feature importance plot to {importance_plot_file}")
    
    return model, test_rmse, test_mae, test_r2

def update_metrics_comparison(test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the hyperparameter-optimized model.
    
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
            f"\n### LightGBM Model with all-MiniLM-L6-v2 embeddings (Hyperopt Tuned, No Count Features)\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < {DIFFICULTY_THRESHOLD} and does not use response count features\n"
            f"- Note: Hyperparameters were optimized using Bayesian optimization with Optuna\n"
        )
        
        # Find the position to insert the new entry (under LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
            
            logger.info("Updated metrics_comparison.txt with hyperopt model results.")
        else:
            logger.error("Could not update metrics_comparison.txt. Section markers not found.")
    
    except Exception as e:
        logger.error(f"Error updating metrics comparison: {str(e)}")

def main():
    """Main function to run the hyperparameter optimization"""
    start_time = time.time()
    
    # Load the data
    df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD)
    
    # Extract features and target
    use_id_features = True
    use_nlp_features = True  # Add NLP-derived features
    
    X, y, feature_names = preprocess_features(
        df, 
        use_id_features=use_id_features,
        use_nlp_features=use_nlp_features
    )
    
    # Create the Optuna study
    study_name = "lightgbm_hyperopt"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    # Optimize hyperparameters
    logger.info(f"Starting hyperparameter optimization with {N_TRIALS} trials...")
    study.optimize(
        partial(objective, X=X, y=y, feature_names=feature_names),
        n_trials=N_TRIALS
    )
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best RMSE: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save best hyperparameters
    with open('hyperopt_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Train the final model with the best hyperparameters
    model, test_rmse, test_mae, test_r2 = train_final_model(
        best_params, X, y, feature_names
    )
    
    # Update metrics comparison file
    update_metrics_comparison(test_rmse, test_mae, test_r2)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final model Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    
    # Check if we beat BERT
    bert_r2 = 0.5300
    if test_r2 > bert_r2:
        logger.info(f"SUCCESS! The optimized LightGBM model (R² = {test_r2:.4f}) outperforms BERT (R² = {bert_r2:.4f})!")
    else:
        logger.info(f"The optimized LightGBM model (R² = {test_r2:.4f}) did not outperform BERT (R² = {bert_r2:.4f}).")
        logger.info("Consider trying additional feature engineering or ensemble methods.")

if __name__ == "__main__":
    main() 
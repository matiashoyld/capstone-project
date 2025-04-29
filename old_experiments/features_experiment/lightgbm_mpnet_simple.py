#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM Model using MPNet Embeddings for Question Difficulty Prediction
This script implements a simpler LightGBM model using MPNet embeddings to predict 
question difficulty, filtering out extremely easy questions (difficulty < -6) 
and avoiding the use of response count features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import ast
import time
import os

# Constants
DIFFICULTY_THRESHOLD = -6
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD):
    """
    Load the question data and filter out questions with difficulty below the threshold.
    
    Args:
        difficulty_threshold: Filter out questions with difficulty below this threshold
        
    Returns:
        DataFrame containing the filtered questions
    """
    print("Loading data...")
    
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
        print(f"Warning: {missing_irt} questions are missing IRT difficulty values")
    
    # Load MPNet embeddings
    with open('questions_mpnet_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge embeddings with questions dataframe
    questions_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    # Filter out questions with difficulty below threshold
    original_count = len(questions_df)
    questions_df = questions_df[questions_df['irt_difficulty'] >= difficulty_threshold]
    filtered_count = original_count - len(questions_df)
    
    print(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
    print(f"Loaded {len(questions_df)} questions with embeddings (after filtering)")
    
    # Print columns to help debugging
    print("\nAvailable columns in the dataset:")
    print(questions_df.columns.tolist())
    
    return questions_df

def preprocess_features(df):
    """
    Preprocess features for model training, focusing on basic features 
    with MPNet embeddings.
    
    Args:
        df: DataFrame containing the question data
        
    Returns:
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
    """
    print("Preprocessing features...")
    
    # Extract embeddings
    embeddings = np.array(df['embedding'].tolist())
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Process categorical features that are available in the dataset
    # Check which columns are available
    available_columns = df_processed.columns.tolist()
    
    # Define potential categorical features to use if available
    potential_categorical = ['subject_id', 'topic_id', 'axis_id', 'complexity', 'skill_level']
    
    # Filter to only use columns that exist in the dataset
    categorical_features = [col for col in potential_categorical if col in available_columns]
    
    print(f"Using categorical features: {categorical_features}")
    
    # Process each available categorical feature
    for feature in categorical_features:
        # Convert to category and fill missing values
        df_processed[feature] = df_processed[feature].fillna('unknown').astype('category')
    
    # Extract only categorical features that are available
    if categorical_features:
        basic_features = df_processed[categorical_features].copy()
        # Convert categorical features to numeric for LightGBM
        for feature in categorical_features:
            basic_features[feature] = basic_features[feature].cat.codes
        
        # Combine embeddings with categorical features
        X = np.hstack([embeddings, basic_features])
        
        # Create feature names
        feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        feature_names.extend(categorical_features)
    else:
        # If no categorical features available, just use embeddings
        X = embeddings
        feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    
    # Extract target variable
    y = df_processed['irt_difficulty'].values
    
    print(f"Processed features shape: {X.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    
    return X, y, feature_names

def train_and_evaluate(X, y, feature_names):
    """
    Train a LightGBM model and evaluate its performance.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Names of features
        
    Returns:
        model: Trained LightGBM model
        cv_scores: Cross-validation scores
        test_rmse: RMSE on test set
        test_mae: MAE on test set
        test_r2: R² on test set
    """
    print("Training and evaluating model...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Basic LightGBM parameters 
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    
    # Train the model
    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=1000)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        # Create dataset for this fold
        cv_train_data = lgb.Dataset(X_cv_train, label=y_cv_train)
        
        # Train model for this fold
        cv_model = lgb.train(params, cv_train_data, num_boost_round=1000)
        
        # Make predictions
        y_cv_pred = cv_model.predict(X_cv_val)
        
        # Calculate RMSE
        cv_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
        cv_scores.append(cv_rmse)
    
    mean_cv_rmse = np.mean(cv_scores)
    std_cv_rmse = np.std(cv_scores)
    print(f"Cross-validation RMSE: {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=20)
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('mpnet_feature_importance.png')
    print("Feature importance plot saved to 'mpnet_feature_importance.png'")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Actual vs Predicted Difficulty')
    plt.tight_layout()
    plt.savefig('mpnet_actual_vs_predicted.png')
    print("Actual vs Predicted plot saved to 'mpnet_actual_vs_predicted.png'")
    
    # Save the model
    model_filename = 'mpnet_lightgbm_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to '{model_filename}'")
    
    return model, cv_scores, test_rmse, test_mae, test_r2

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
        "MPNet LightGBM Simple": {
            "Mean CV RMSE": f"{mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}",
            "Test RMSE": f"{test_rmse:.4f}",
            "Test MAE": f"{test_mae:.4f}",
            "Test R²": f"{test_r2:.4f}"
        }
    }
    
    # Create or update comparison file
    comparison_file = 'model_metrics_comparison.json'
    
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
    else:
        comparison = {}
    
    # Update with new metrics
    comparison.update(metrics)
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"Metrics comparison updated in '{comparison_file}'")

def main():
    """Main function to run the complete model pipeline."""
    start_time = time.time()
    
    # Load and preprocess data
    df = load_data()
    X, y, feature_names = preprocess_features(df)
    
    # Train and evaluate model
    model, cv_scores, test_rmse, test_mae, test_r2 = train_and_evaluate(X, y, feature_names)
    
    # Update metrics comparison
    update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2)
    
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    
    # Print summary of results for easy comparison
    print("\n===== MPNet LightGBM Model Results =====")
    print(f"Cross-validation RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print("========================================")

if __name__ == "__main__":
    main() 
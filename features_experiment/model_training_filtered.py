#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM Model for Question Difficulty Prediction
This script implements a LightGBM model approach to predict question difficulty,
filtering out extremely easy questions (difficulty < -6).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import ast  # For safely evaluating the skills string representation

def load_data(difficulty_threshold=-6):
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
    
    # Load embeddings
    with open('questions_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge embeddings with questions dataframe
    questions_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    # Filter out questions with difficulty below threshold
    original_count = len(questions_df)
    questions_df = questions_df[questions_df['irt_difficulty'] >= difficulty_threshold]
    filtered_count = original_count - len(questions_df)
    
    print(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
    print(f"Loaded {len(questions_df)} questions with embeddings (after filtering)")
    
    # Display a sample of the data
    print("\nSample of the loaded data:")
    print(questions_df.head())
    
    # Display data info
    print("\nData info:")
    print(questions_df.info())
    
    return questions_df

def preprocess_features(df, use_id_features=True):
    """
    Preprocess features for model training.
    
    Args:
        df: DataFrame containing the question data
        use_id_features: Whether to use ID features (topic_id, subject_id, axis_id)
        
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
    
    # Process skills
    mlb = MultiLabelBinarizer()
    skills_lists = df_processed['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    skills_binary = mlb.fit_transform(skills_lists)
    skills_df = pd.DataFrame(skills_binary, columns=mlb.classes_)
    
    # Convert all column names to strings
    skills_df.columns = skills_df.columns.astype(str)
    
    # Basic features
    basic_features = ['level', 'num_misconceptions', 'has_image', 'count_a', 'count_b', 
                      'count_c', 'count_d', 'count_e', 'total_count', 'avg_steps']
    
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
    
    with open('feature_processors_filtered.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X, y, feature_names

def train_and_evaluate(X, y, feature_names, use_tuning=False):
    """
    Train and evaluate the LightGBM model.
    
    Args:
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        use_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        model: Trained LightGBM model
        cv_scores: Cross-validation scores
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    print("Training and evaluating model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Default parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    # No hyperparameter tuning for this filtered version
    print("Hyperparameter tuning skipped.")
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        lgb_train = lgb.Dataset(X_cv_train, y_cv_train)
        lgb_val = lgb.Dataset(X_cv_val, y_cv_val, reference=lgb_train)
        
        # Remove n_estimators from params for LightGBM
        train_params = params.copy()
        if 'n_estimators' in train_params:
            del train_params['n_estimators']
        
        # Set fixed num_boost_round instead of n_estimators
        num_boost_round = 100
        
        model = lgb.train(
            train_params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        y_pred = model.predict(X_cv_val)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_pred))
        cv_scores.append(rmse)
    
    print(f"Cross-validation RMSE scores: {cv_scores}")
    print(f"Mean CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all training data
    lgb_train = lgb.Dataset(X_train, y_train)
    
    # Remove n_estimators from params for LightGBM
    train_params = params.copy()
    if 'n_estimators' in train_params:
        del train_params['n_estimators']
    
    # Set fixed num_boost_round instead of n_estimators
    num_boost_round = 100
    
    model = lgb.train(
        train_params,
        lgb_train,
        num_boost_round=num_boost_round
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save model
    with open('question_model_filtered.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Create and save scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Actual vs Predicted Difficulty (Filtered Model)')
    plt.savefig('prediction_scatter_filtered.png')
    
    return model, cv_scores, test_rmse, test_mae, test_r2

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance.
    
    Args:
        model: Trained LightGBM model
        feature_names: Names of the processed features
    """
    print("Analyzing feature importance...")
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_filtered.csv', index=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns_plot = plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Filtered Model)')
    plt.tight_layout()
    plt.savefig('feature_importance_filtered.png')

def update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the filtered model.
    
    Args:
        cv_scores: Cross-validation scores
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
            
        # Create new entry for filtered model
        new_entry = (
            f"\n### LightGBM Model with all-MiniLM-L6-v2 embeddings (Filtered: difficulty >= -6)\n"
            f"- Mean CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < -6\n"
        )
        
        # Find the position to insert the new entry (under LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
                
            print("Updated metrics_comparison.txt with filtered model results.")
        else:
            print("Could not update metrics_comparison.txt. Section markers not found.")
            
    except Exception as e:
        print(f"Error updating metrics comparison: {str(e)}")

def main():
    """Main function to run the filtered model training pipeline."""
    
    # Load the data and display a sample and data info
    df = load_data(difficulty_threshold=-6)
    
    # Check that we have the expected target variable
    if 'irt_difficulty' in df.columns:
        # Print statistics for the target variable
        target = 'irt_difficulty'
        target_stats = df[target].describe()
        print(f"\nTarget variable statistics after filtering:")
        print(f"Min: {target_stats['min']:.4f}, Max: {target_stats['max']:.4f}, "
              f"Mean: {target_stats['mean']:.4f}, Std: {target_stats['std']:.4f}")
        
        # Process features
        X, y, feature_names = preprocess_features(df, use_id_features=True)
        
        # Skip hyperparameter tuning
        use_tuning = False
        
        # Train and evaluate model
        model, cv_scores, test_rmse, test_mae, test_r2 = train_and_evaluate(
            X, y, feature_names, use_tuning=use_tuning
        )
        
        # Analyze feature importance
        analyze_feature_importance(model, feature_names)
        
        # Update metrics comparison file
        update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2)
        
        print("\nModel training completed. Files saved:")
        print("- question_model_filtered.pkl: The trained model")
        print("- feature_processors_filtered.pkl: Feature processing objects")
        print("- feature_importance_filtered.csv: Feature importance data")
        print("- feature_importance_filtered.png: Visualization of feature importance")
        print("- prediction_scatter_filtered.png: Scatter plot of actual vs predicted values")
        
    else:
        print(f"Error: Target column '{target}' not found in the dataframe.")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import seaborn as sns
    
    main() 
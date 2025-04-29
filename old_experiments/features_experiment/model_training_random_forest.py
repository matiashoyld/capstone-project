#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forest Model for Question Difficulty Prediction (Without Response Counts)
This script implements a Random Forest model approach to predict question difficulty,
filtering out extremely easy questions (difficulty < -6) and avoiding the use of
response count features that would not be available for new questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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
    
    # Basic features without count features
    basic_features = ['level', 'num_misconceptions', 'has_image', 'avg_steps']
    
    # Removed count features: 'count_a', 'count_b', 'count_c', 'count_d', 'count_e', 'total_count'
    print("Note: Removed response count features which wouldn't be available for new questions")
    
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
    
    with open('feature_processors_random_forest.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X, y, feature_names

def train_and_evaluate(X, y, feature_names, use_tuning=False):
    """
    Train and evaluate the Random Forest model.
    
    Args:
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        use_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        model: Trained Random Forest model
        cv_scores: Cross-validation scores
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    print("Training and evaluating Random Forest model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Default parameters for Random Forest
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1  # Use all available cores
    }
    
    # No hyperparameter tuning for this version
    print("Hyperparameter tuning skipped.")
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_cv_train, y_cv_train)
        
        # Evaluate
        y_pred = model.predict(X_cv_val)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_pred))
        cv_scores.append(rmse)
    
    print(f"Cross-validation RMSE scores: {cv_scores}")
    print(f"Mean CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all training data
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save model
    with open('question_model_random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Create and save scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Actual vs Predicted Difficulty (Random Forest, No Count Features)')
    plt.savefig('prediction_scatter_random_forest.png')
    
    return model, cv_scores, test_rmse, test_mae, test_r2

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance.
    
    Args:
        model: Trained Random Forest model
        feature_names: Names of the processed features
    """
    print("Analyzing feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_random_forest.csv', index=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Random Forest, No Count Features)')
    plt.tight_layout()
    plt.savefig('feature_importance_random_forest.png')

def update_metrics_comparison(cv_scores, test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the Random Forest model.
    
    Args:
        cv_scores: Cross-validation scores
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
            
        # Create new entry for Random Forest model
        new_entry = (
            f"\n### Random Forest Model with all-MiniLM-L6-v2 embeddings (Filtered: difficulty >= -6, No Count Features)\n"
            f"- Mean CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < -6 and does not use response count features\n"
        )
        
        # Find the position to insert the new entry (after LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
                
            print("Updated metrics_comparison.txt with Random Forest model results.")
        else:
            # If we can't find the sections, append to the end
            with open('metrics_comparison.txt', 'a') as f:
                f.write("\n## Random Forest Models\n")
                f.write(new_entry)
            print("Added Random Forest section to metrics_comparison.txt")
            
    except Exception as e:
        print(f"Error updating metrics comparison: {str(e)}")
        # Create a new file if it doesn't exist
        try:
            with open('metrics_comparison.txt', 'w') as f:
                f.write("# Model Performance Comparison\n\n")
                f.write("## Random Forest Models\n")
                f.write(new_entry)
            print("Created new metrics_comparison.txt with Random Forest model results.")
        except Exception as e2:
            print(f"Failed to create metrics_comparison.txt: {str(e2)}")

def main():
    """Main function to run the Random Forest model training pipeline."""
    
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
        print("- question_model_random_forest.pkl: The trained model")
        print("- feature_processors_random_forest.pkl: Feature processing objects")
        print("- feature_importance_random_forest.csv: Feature importance data")
        print("- feature_importance_random_forest.png: Visualization of feature importance")
        print("- prediction_scatter_random_forest.png: Scatter plot of actual vs predicted values")
        
    else:
        print(f"Error: Target column '{target}' not found in the dataframe.")

if __name__ == "__main__":
    main() 
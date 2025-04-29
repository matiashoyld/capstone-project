#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Question Difficulty Prediction Model Training Script
This script loads question data and embeddings, processes features, and trains an ML model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pickle
import json
import ast  # For safely evaluating the skills string representation
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load and merge the question data and embeddings"""
    print("Loading data...")
    
    # Load questions data
    questions_df = pd.read_csv('questions_master.csv')

    new_irt_difficulties = pd.read_csv('question_difficulties_irt.csv')

    # Merge IRT difficulties into questions dataframe
    questions_df = questions_df.merge(new_irt_difficulties[['question_id', 'irt_difficulty']], 
                                      on='question_id', 
                                      how='left')
    
    # Check if any questions are missing IRT difficulty values
    missing_irt = questions_df['irt_difficulty'].isna().sum()
    if missing_irt > 0:
        print(f"Warning: {missing_irt} questions are missing IRT difficulty values")
    
    # Load embeddings
    embeddings_df = pd.read_pickle('questions_embeddings.pkl')
    
    # Merge dataframes on question_id
    merged_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    print(f"Loaded {len(merged_df)} questions with embeddings")
    return merged_df

def preprocess_features(df):
    """Process all features for model training"""
    print("Preprocessing features...")
    
    # Convert skills from string representation to actual lists if needed
    if isinstance(df['skills'].iloc[0], str):
        df['skills'] = df['skills'].apply(ast.literal_eval)
    
    # Process numeric features
    numeric_features = ['avg_steps', 'num_misconceptions', 'level']  # Updated column names
    X_numeric = df[numeric_features].copy()
    
    # Process ID features (one-hot encoding)
    id_features = ['topic_id', 'subject_id', 'axis_id']
    X_ids = df[id_features].copy()
    
    # Handle missing values
    X_numeric.fillna(0, inplace=True)
    X_ids.fillna(-1, inplace=True)  # Use -1 for missing IDs
    
    # One-hot encode the ID features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ids_encoded = encoder.fit_transform(X_ids)
    id_feature_names = []
    for i, feature in enumerate(id_features):
        for j in range(len(encoder.categories_[i])):
            id_feature_names.append(f"{feature}_{encoder.categories_[i][j]}")
    X_ids_encoded = pd.DataFrame(X_ids_encoded, columns=id_feature_names)
    
    # Normalize numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_features)
    
    # Process binary features
    X_binary = df[['has_image']].astype(int)
    
    # Process skills (one-hot encoding)
    mlb = MultiLabelBinarizer()
    X_skills = mlb.fit_transform(df['skills'])
    skill_cols = [f'skill_{i}' for i in mlb.classes_]
    X_skills_df = pd.DataFrame(X_skills, columns=skill_cols)
    
    # Process embeddings
    X_embeddings = np.array(df['embedding'].tolist())
    
    # Save the feature processors for later use
    with open('feature_processors.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'mlb': mlb,
            'encoder': encoder,
        }, f)
    
    # Combine all processed features
    X_combined = pd.concat([
        X_numeric_scaled, 
        X_binary,
        X_skills_df,
        X_ids_encoded  # Add one-hot encoded ID features
    ], axis=1)
    
    # Return processed features and the embedding array separately
    # This allows flexibility in how we use the embeddings
    return X_combined, X_embeddings, mlb.classes_

def hyperparameter_tuning(X_train, y_train, cv=5, n_iter=20):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values
    cv : int
        Number of cross-validation folds
    n_iter : int
        Number of parameter settings sampled
        
    Returns:
    --------
    best_params : dict
        Dictionary of best hyperparameters
    best_score : float
        Best score achieved
    """
    print("Performing hyperparameter tuning...")
    
    # Define the parameter space
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, -1],  # -1 means no limit
        'num_leaves': [31, 63, 127, 255],
        'min_child_samples': [5, 10, 20, 50],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    # Initialize LightGBM regressor
    model = lgb.LGBMRegressor(random_state=42, 
                             objective='regression', 
                             verbose=-1)
    
    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv,
        random_state=42,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit RandomizedSearchCV
    search.fit(X_train, y_train)
    
    # Get best parameters and score
    best_params = search.best_params_
    best_score = np.sqrt(-search.best_score_)  # Convert neg MSE to RMSE
    
    print(f"Best RMSE: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params, best_score

def train_model(X, y, embeddings=None, cv=5, use_best_params=False):
    """Train and evaluate the regression model"""
    print("Training model...")
    
    # Combine structured features with embeddings
    if embeddings is not None:
        X_with_embeddings = np.hstack([X.values, embeddings])
    else:
        X_with_embeddings = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_embeddings, y, test_size=0.2, random_state=42
    )
    
    # Get best parameters if requested
    if use_best_params:
        print("Performing hyperparameter tuning...")
        best_params, _ = hyperparameter_tuning(X_train, y_train, cv=cv)
        print(f"Using tuned hyperparameters: {best_params}")
        model = lgb.LGBMRegressor(**best_params, random_state=42)
    else:
        # Use default parameters
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42
        )
    
    # Cross-validation for regression (using negative MSE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_cv_scores = np.sqrt(-cv_scores)  # Convert negative MSE to RMSE
    print(f"Cross-validation RMSE scores: {rmse_cv_scores}")
    print(f"Mean CV RMSE: {rmse_cv_scores.mean():.4f} ± {rmse_cv_scores.std():.4f}")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create a plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([-11, 3], [-11, 3], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Actual vs Predicted Difficulty')
    plt.grid(True)
    plt.savefig('prediction_scatter.png')
    
    # Save model
    with open('question_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_train, X_test, y_train, y_test

def analyze_feature_importance(model, X, skill_classes, embeddings_used=True):
    """Analyze and visualize feature importance"""
    print("Analyzing feature importance...")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Create feature names
    if embeddings_used:
        # If embeddings were used, we need to account for them in feature names
        embedding_cols = [f'emb_{i}' for i in range(768)]  # 768 is the dimension of the embeddings
        feature_names = list(X.columns) + embedding_cols
    else:
        feature_names = list(X.columns)
    
    # Map skill IDs to more readable names
    readable_names = []
    for name in feature_names:
        if name.startswith('skill_'):
            skill_id = name.replace('skill_', '')
            readable_names.append(f'Skill {skill_id}')
        else:
            readable_names.append(name)
    
    # Create a DataFrame for importance
    importance_df = pd.DataFrame({
        'feature': readable_names[:len(feature_importance)],
        'importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance for Difficulty Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save feature importance to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    
    return importance_df

def main():
    # Load the data
    df = load_data()
    
    print("\nSample of the loaded data:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())
    
    # Define target variable as difficulty (continuous)
    target_col = 'irt_difficulty'
    if target_col in df.columns:
        y = df[target_col]
        
        print(f"\nTarget variable statistics:")
        print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        
        # Process features
        X_structured, X_embeddings, skill_classes = preprocess_features(df)
        
        # Set use_tuning to False to skip hyperparameter tuning
        use_tuning = False
        print("\nSkipping hyperparameter tuning as requested.")
        
        # Train and evaluate model
        model, X_train, X_test, y_train, y_test = train_model(
            X_structured, y, X_embeddings, use_best_params=use_tuning
        )
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(model, X_structured, skill_classes)
        
        print("\nModel training completed. Files saved:")
        print("- question_model.pkl: The trained model")
        print("- feature_processors.pkl: Feature processing objects")
        print("- feature_importance.csv: Feature importance data")
        print("- feature_importance.png: Feature importance visualization")
        if use_tuning:
            print("- best_params.json: Best hyperparameters from tuning")
        print("- prediction_scatter.png: Scatter plot of actual vs predicted values")
    else:
        print(f"Error: Target column '{target_col}' not found in data.")
        print(f"Available columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main() 
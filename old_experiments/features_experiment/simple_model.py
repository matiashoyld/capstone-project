#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Question Difficulty Prediction Model (No Embeddings)
This script is similar to model_training.py but trains a model without using text embeddings.
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
    """Load the question data (without embeddings)"""
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
    
    print(f"Loaded {len(questions_df)} questions")
    return questions_df

def preprocess_features(df):
    """Process all features for model training"""
    print("Preprocessing features...")
    
    # Convert skills from string representation to actual lists if needed
    if isinstance(df['skills'].iloc[0], str):
        df['skills'] = df['skills'].apply(ast.literal_eval)
    
    # Process numeric features
    numeric_features = ['avg_steps', 'num_misconceptions', 'level']  # Updated column names
    X_numeric = df[numeric_features].copy()
    
    # Add ID features - topic_id, subject_id, and axis_id
    id_features = ['topic_id', 'subject_id', 'axis_id']
    X_ids = df[id_features].copy()
    
    # Handle missing values
    X_numeric.fillna(0, inplace=True)
    X_ids.fillna(-1, inplace=True)  # Use -1 for missing IDs
    
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
    
    # Save the feature processors for later use
    with open('feature_processors_simple.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'mlb': mlb,
        }, f)
    
    # Combine all processed features
    X_combined = pd.concat([
        X_numeric_scaled, 
        X_binary,
        X_skills_df,
        X_ids  # Add ID features
    ], axis=1)
    
    return X_combined, mlb.classes_

def hyperparameter_tuning(X_train, y_train, cv=5, n_iter=20):
    """Perform hyperparameter tuning for LightGBM model"""
    print("Performing hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': np.arange(50, 500, 50),
        'learning_rate': np.logspace(-3, 0, 20),
        'max_depth': np.arange(3, 12),
        'num_leaves': np.arange(10, 100, 10),
        'min_child_samples': np.arange(5, 50, 5),
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5),
        'reg_alpha': np.logspace(-3, 1, 10),
        'reg_lambda': np.logspace(-3, 1, 10)
    }
    
    # Initialize LightGBM model
    model = lgb.LGBMRegressor(random_state=42)
    
    # Initialize random search
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit random search
    search.fit(X_train, y_train)
    
    # Get best parameters and score
    best_params = search.best_params_
    best_score = np.sqrt(-search.best_score_)  # Convert neg MSE to RMSE
    
    print(f"Best RMSE: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Save best parameters
    with open('best_params_simple.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params, best_score

def train_model(X, y, cv=5, use_best_params=False):
    """Train and evaluate the regression model (without embeddings)"""
    print("Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    plt.title('Actual vs Predicted Difficulty (No Embeddings)')
    plt.grid(True)
    plt.savefig('prediction_scatter_simple.png')
    
    # Save model
    with open('question_model_simple.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_train, X_test, y_train, y_test

def analyze_feature_importance(model, X, skill_classes):
    """Analyze and visualize feature importance"""
    print("Analyzing feature importance...")
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create DataFrame with feature importance
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Save to CSV
    importance_df.to_csv('feature_importance_simple.csv', index=False)
    
    # Plot feature importance (top 20)
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importance (No Embeddings)')
    plt.tight_layout()
    plt.savefig('feature_importance_simple.png')
    
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
        
        # Process features without embeddings
        X_structured, skill_classes = preprocess_features(df)
        
        # Set use_tuning to False to skip hyperparameter tuning
        use_tuning = False
        print("\nSkipping hyperparameter tuning as requested.")
        
        # Train and evaluate model
        model, X_train, X_test, y_train, y_test = train_model(
            X_structured, y, use_best_params=use_tuning
        )
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(model, X_structured, skill_classes)
        
        print("\nModel training completed. Files saved:")
        print("- question_model_simple.pkl: The trained model")
        print("- feature_processors_simple.pkl: Feature processing objects")
        print("- feature_importance_simple.csv: Feature importance data")
        print("- feature_importance_simple.png: Feature importance visualization")
        if use_tuning:
            print("- best_params_simple.json: Best hyperparameters from tuning")
        print("- prediction_scatter_simple.png: Scatter plot of actual vs predicted values")
    else:
        print(f"Error: Target column '{target_col}' not found in data.")
        print(f"Available columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main() 
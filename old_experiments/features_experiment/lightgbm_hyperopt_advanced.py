#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced LightGBM Hyperparameter Optimization with Enhanced Feature Engineering

This script performs advanced hyperparameter tuning for the LightGBM model
using Optuna for Bayesian optimization, with additional feature engineering
techniques to improve performance beyond BERT's R² of 0.5300.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import json
import ast  # For safely evaluating the skills string representation
import logging
import time
import re
from functools import partial
import nltk
from nltk.corpus import stopwords
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 embeddings
PCA_COMPONENTS = 50  # Number of PCA components for embeddings
N_CLUSTERS = 8  # Number of clusters for K-means clustering

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
            'mathematical_symbols': 0
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
    
    # Count mathematical symbols
    math_symbols = set(['+', '-', '*', '/', '=', '<', '>', '±', '≤', '≥', '≠', '≈', '∞', '∫', '∑', '∏', '√', '^'])
    mathematical_symbols = sum(text.count(sym) for sym in math_symbols)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'mathematical_symbols': mathematical_symbols
    }

def create_difficulty_bins(y, n_bins=10):
    """
    Create bins of difficulty for stratified cross-validation.
    
    Args:
        y: Target variable (difficulty scores)
        n_bins: Number of bins to create
        
    Returns:
        Array of bin indices for each data point
    """
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

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

def preprocess_features(df, use_id_features=True, use_nlp_features=True, 
                        use_advanced_features=True, reduce_embeddings=True):
    """
    Preprocess features for model training with advanced feature engineering.
    
    Args:
        df: DataFrame containing the question data
        use_id_features: Whether to use ID features (topic_id, subject_id, axis_id)
        use_nlp_features: Whether to use basic NLP-derived features
        use_advanced_features: Whether to use advanced text features and transformations
        reduce_embeddings: Whether to reduce dimensionality of embeddings
        
    Returns:
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
    """
    logger.info("Preprocessing features with advanced engineering...")
    
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
    
    # Add computed basic text-based features
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
    
    # Add advanced text features
    if use_advanced_features:
        logger.info("Adding advanced text features...")
        
        # Extract text features from question title
        text_features = df_processed['question_title'].apply(
            lambda x: extract_text_features(x) if pd.notna(x) else extract_text_features("")
        )
        
        # Convert to DataFrame
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Interaction features
        df_processed['math_to_text_ratio'] = text_features_df['mathematical_symbols'] / (text_features_df['word_count'] + 1)
        df_processed['digit_to_text_ratio'] = text_features_df['digit_count'] / (text_features_df['word_count'] + 1)
        df_processed['special_to_text_ratio'] = text_features_df['special_char_count'] / (text_features_df['word_count'] + 1)
        
        # Add text features to processed dataframe
        for col in text_features_df.columns:
            df_processed[col] = text_features_df[col]
        
        # Create subject-level interaction
        df_processed['subject_level'] = df_processed['subject_id'].astype(str) + '_' + df_processed['level'].astype(str)
        
        # K-means clustering on embeddings
        logger.info("Performing K-means clustering on embeddings...")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df_processed['embedding_cluster'] = kmeans.fit_predict(embeddings)
        
        # Create TF-IDF features from question text
        logger.info("Generating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=100, 
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
        
        # Add these additional features to basic features
        advanced_features = [
            'word_count', 'char_count', 'avg_word_length', 
            'digit_count', 'special_char_count', 'mathematical_symbols',
            'math_to_text_ratio', 'digit_to_text_ratio', 'special_to_text_ratio',
            'embedding_cluster'
        ]
        basic_features.extend(advanced_features)
    else:
        tfidf_df = pd.DataFrame()
    
    logger.info("Using only features available for new questions (no count features)")
    
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
        
        # Add subject-level one-hot encoding if using advanced features
        if use_advanced_features:
            subject_level_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            subject_level_encoded = subject_level_encoder.fit_transform(
                df_processed[['subject_level']]
            )
            subject_level_cols = [f"subject_level_{i}" for i in range(subject_level_encoded.shape[1])]
            subject_level_df = pd.DataFrame(subject_level_encoded, columns=subject_level_cols)
            id_features_df = pd.concat([id_features_df, subject_level_df], axis=1)
            id_features.extend(subject_level_cols)
    else:
        id_features_df = pd.DataFrame()
    
    # Combine features
    feature_dfs = [df_processed[basic_features].reset_index(drop=True),
                   skills_df.reset_index(drop=True)]
    
    if use_id_features and not id_features_df.empty:
        feature_dfs.append(id_features_df.reset_index(drop=True))
    
    if use_advanced_features and not tfidf_df.empty:
        feature_dfs.append(tfidf_df.reset_index(drop=True))
    
    X_without_embeddings = pd.concat(feature_dfs, axis=1)
    
    # Convert all column names to strings to avoid validation errors
    X_without_embeddings.columns = X_without_embeddings.columns.astype(str)
    
    # Apply power transformation to improve normality
    if use_advanced_features:
        logger.info("Applying power transformation to features...")
        numeric_cols = X_without_embeddings.select_dtypes(include=['number']).columns
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_without_embeddings_transformed = X_without_embeddings.copy()
        X_without_embeddings_transformed[numeric_cols] = pt.fit_transform(X_without_embeddings[numeric_cols])
        
        # Replace the original dataframe with the transformed one
        X_without_embeddings = X_without_embeddings_transformed
    else:
        # Scale features using standard scaler
        scaler = StandardScaler()
        X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings)
        X_without_embeddings = pd.DataFrame(X_without_embeddings_scaled, columns=X_without_embeddings.columns)
        pt = None
    
    # Apply dimensionality reduction to embeddings if requested
    if reduce_embeddings and use_advanced_features:
        logger.info(f"Reducing embedding dimensionality with PCA to {PCA_COMPONENTS} components...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        embeddings_reduced = pca.fit_transform(embeddings)
        logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Feature names for reduced embeddings
        embedding_feature_names = [f"pca_emb_{i}" for i in range(embeddings_reduced.shape[1])]
    else:
        embeddings_reduced = embeddings
        embedding_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        pca = None
    
    # Add embeddings to processed features (as NumPy array)
    X_array = np.hstack([X_without_embeddings.values, embeddings_reduced])
    
    # Feature names (for feature importance)
    feature_names = list(X_without_embeddings.columns) + embedding_feature_names
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
    }
    
    if use_id_features:
        processors['id_encoder'] = encoder
        if use_advanced_features:
            processors['subject_level_encoder'] = subject_level_encoder
    
    if use_advanced_features:
        processors['power_transformer'] = pt
        processors['kmeans'] = kmeans
        processors['tfidf'] = tfidf
        if reduce_embeddings:
            processors['pca'] = pca
    
    with open('feature_processors_advanced.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X_array, y, feature_names

def objective(trial, X, y, feature_names, difficulty_bins):
    """
    Objective function for Optuna optimization with stratified k-fold.
    
    Args:
        trial: Optuna trial object
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        difficulty_bins: Binned difficulty scores for stratification
        
    Returns:
        Mean validation RMSE across folds
    """
    # Define the hyperparameter search space
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 200),
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
    
    # Add dart-specific parameters if boosting_type is 'dart'
    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.5)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.05, 0.5)
    
    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    r2_scores = []
    
    for train_idx, val_idx in skf.split(X, difficulty_bins):
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
                lgb.early_stopping(stopping_rounds=100, verbose=False),
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
    
    # Split data into train and test sets (stratified by binned difficulty)
    difficulty_bins = create_difficulty_bins(y)
    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        X, y, difficulty_bins, test_size=test_size, random_state=RANDOM_STATE, stratify=difficulty_bins
    )
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train final model
    num_boost_round = 10000
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
    model_file = 'question_model_advanced.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved optimized model to {model_file}")
    
    # Create and save scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Advanced LightGBM: Actual vs Predicted Difficulty')
    scatter_file = 'prediction_scatter_advanced.png'
    plt.savefig(scatter_file)
    plt.close()
    logger.info(f"Saved prediction scatter plot to {scatter_file}")
    
    # Analyze feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_file = 'feature_importance_advanced.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"Saved feature importance to {importance_file}")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Advanced Model)')
    plt.tight_layout()
    importance_plot_file = 'feature_importance_advanced.png'
    plt.savefig(importance_plot_file)
    plt.close()
    logger.info(f"Saved feature importance plot to {importance_plot_file}")
    
    # Analyze error distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    error_file = 'error_distribution_advanced.png'
    plt.savefig(error_file)
    plt.close()
    logger.info(f"Saved error distribution to {error_file}")
    
    # Error analysis by difficulty range
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, np.abs(errors), alpha=0.5)
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs. Actual Difficulty')
    plt.grid(True, alpha=0.3)
    error_analysis_file = 'error_analysis_advanced.png'
    plt.savefig(error_analysis_file)
    plt.close()
    logger.info(f"Saved error analysis to {error_analysis_file}")
    
    return model, test_rmse, test_mae, test_r2

def update_metrics_comparison(test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the advanced model.
    
    Args:
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
        
        # Create new entry for advanced model
        new_entry = (
            f"\n### LightGBM Model with Advanced Features (No Count Features)\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < {DIFFICULTY_THRESHOLD} and does not use response count features\n"
            f"- Note: Uses advanced feature engineering and stratified sampling\n"
        )
        
        # Find the position to insert the new entry (under LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
            
            logger.info("Updated metrics_comparison.txt with advanced model results.")
        else:
            logger.error("Could not update metrics_comparison.txt. Section markers not found.")
    
    except Exception as e:
        logger.error(f"Error updating metrics comparison: {str(e)}")

def main():
    """Main function to run the hyperparameter optimization with advanced features"""
    start_time = time.time()
    
    # Load the data
    df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD)
    
    # Extract features and target with advanced feature engineering
    X, y, feature_names = preprocess_features(
        df, 
        use_id_features=True,
        use_nlp_features=True,
        use_advanced_features=True,
        reduce_embeddings=True
    )
    
    # Create difficulty bins for stratified sampling
    difficulty_bins = create_difficulty_bins(y)
    
    # Create the Optuna study
    study_name = "lightgbm_advanced"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    # Optimize hyperparameters
    logger.info(f"Starting hyperparameter optimization with {N_TRIALS} trials...")
    study.optimize(
        partial(objective, X=X, y=y, feature_names=feature_names, difficulty_bins=difficulty_bins),
        n_trials=N_TRIALS
    )
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best RMSE: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save best hyperparameters
    with open('hyperopt_best_params_advanced.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Train the final model with the best hyperparameters
    model, test_rmse, test_mae, test_r2 = train_final_model(
        best_params, X, y, feature_names
    )
    
    # Update metrics comparison file
    update_metrics_comparison(test_rmse, test_mae, test_r2)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    logger.info(f"Advanced hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final model Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    
    # Check if we beat BERT
    bert_r2 = 0.5300
    if test_r2 > bert_r2:
        logger.info(f"SUCCESS! The advanced LightGBM model (R² = {test_r2:.4f}) outperforms BERT (R² = {bert_r2:.4f})!")
    else:
        logger.info(f"The advanced LightGBM model (R² = {test_r2:.4f}) did not outperform BERT (R² = {bert_r2:.4f}).")
        logger.info("Consider trying ensemble methods or blending with BERT predictions.")

if __name__ == "__main__":
    main() 
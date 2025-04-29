#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced LightGBM Hyperparameter Optimization with Fast Execution

This script combines advanced feature engineering techniques with faster
hyperparameter optimization to quickly produce a high-performance model
that can potentially outperform BERT.
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
from sklearn.decomposition import PCA
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
from tqdm import tqdm  # Add tqdm for progress tracking

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
N_TRIALS = 20  # Reduced from 30 to 20 trials
CV_FOLDS = 3  # Reduced number of folds
RANDOM_STATE = 42  # For reproducibility
SAMPLE_FRACTION = 0.3  # Reduce from 0.4 to 0.3 for faster execution
N_JOBS = -1  # Number of parallel jobs (-1 means use all available cores)
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 embeddings
PCA_COMPONENTS = 50  # Number of PCA components for embeddings
N_CLUSTERS = 8  # Number of clusters for K-means clustering
TFIDF_FEATURES = 50  # Number of TF-IDF features to use

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
        logger.info(f"Sampled {len(questions_df)} questions ({sample_fraction:.0%} of filtered data) for faster tuning")
    
    return questions_df

def preprocess_features(df, use_advanced_features=True, reduce_embeddings=True):
    """
    Preprocess features for model training with advanced feature engineering.
    
    Args:
        df: DataFrame containing the question data
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
    
    # Add computed text-based features
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
            max_features=TFIDF_FEATURES, 
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
    
    logger.info("Note: Using only features available for new questions (no count features)")
    
    # ID features
    # One-hot encode categorical ID features
    id_cols = ['topic_id', 'subject_id', 'axis_id']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_ids = encoder.fit_transform(df_processed[id_cols])
    id_feature_names = [f"{col}_{val}" for col, vals in zip(id_cols, encoder.categories_) 
                        for val in vals]
    id_features_df = pd.DataFrame(encoded_ids, columns=id_feature_names)
    
    # Add subject-level one-hot encoding if using advanced features
    if use_advanced_features:
        subject_level_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        subject_level_encoded = subject_level_encoder.fit_transform(
            df_processed[['subject_level']]
        )
        subject_level_cols = [f"subject_level_{i}" for i in range(subject_level_encoded.shape[1])]
        subject_level_df = pd.DataFrame(subject_level_encoded, columns=subject_level_cols)
        id_features_df = pd.concat([id_features_df, subject_level_df], axis=1)
    
    # Combine features
    feature_dfs = [df_processed[basic_features].reset_index(drop=True),
                   skills_df.reset_index(drop=True),
                   id_features_df.reset_index(drop=True)]
    
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
        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"Explained variance ratio: {explained_variance:.4f}")
        
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
        'id_encoder': encoder,
    }
    
    if use_advanced_features:
        processors['power_transformer'] = pt
        processors['kmeans'] = kmeans
        processors['tfidf'] = tfidf
        processors['subject_level_encoder'] = subject_level_encoder
        if reduce_embeddings:
            processors['pca'] = pca
    
    with open('feature_processors_enhanced.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X_array, y, feature_names

def objective(trial, X, y, feature_names):
    """
    Objective function for Optuna optimization with stratified k-fold.
    
    Args:
        trial: Optuna trial object
        X: Processed features
        y: Target variable
        feature_names: Names of the processed features
        
    Returns:
        Mean validation RMSE across folds
    """
    logger.info(f"Starting trial {trial.number} of {N_TRIALS}")
    
    # Define a focused hyperparameter search space with narrower ranges
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),  # Focus on gbdt as most stable
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'verbose': -1,
    }
    
    logger.info(f"Trial {trial.number} parameters: {param}")
    
    # Stratified k-fold cross-validation using binned difficulty values
    difficulty_bins = create_difficulty_bins(y)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, difficulty_bins)):
        logger.info(f"  Trial {trial.number}, Fold {fold_idx+1}/{CV_FOLDS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with fewer rounds for speed
        max_boost_round = 1000
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
        trial.report(model.best_score['valid_0']['rmse'], step=fold_idx)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            logger.info(f"  Trial {trial.number} pruned at fold {fold_idx+1}")
            raise optuna.exceptions.TrialPruned()
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        logger.info(f"  Fold {fold_idx+1} RMSE: {rmse:.4f}, R²: {r2:.4f}")
        rmse_scores.append(rmse)
    
    mean_rmse = np.mean(rmse_scores)
    
    # Log metrics
    logger.info(f"Trial {trial.number} completed: RMSE = {mean_rmse:.4f}")
    
    # Return the mean RMSE (objective to minimize)
    return mean_rmse

def train_full_model(best_params, df_full, test_size=0.2):
    """
    Train the final LightGBM model on the full dataset with the best hyperparameters.
    
    Args:
        best_params: Best hyperparameters found by Optuna
        df_full: Full DataFrame with question data
        test_size: Proportion of data to use for testing
        
    Returns:
        model: Trained LightGBM model
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
        feature_names: Names of the features used
    """
    logger.info("Training final model on the full dataset with best hyperparameters...")
    
    # Preprocess the full dataset with all advanced features
    X_full, y_full, feature_names = preprocess_features(
        df_full, 
        use_advanced_features=True,
        reduce_embeddings=False  # Use all embedding dimensions for final model
    )
    
    # Split data into train and test sets (stratified by binned difficulty)
    difficulty_bins = create_difficulty_bins(y_full)
    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        X_full, y_full, difficulty_bins, test_size=test_size, random_state=RANDOM_STATE, stratify=difficulty_bins
    )
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train final model with more boosting rounds
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
    model_file = 'question_model_enhanced.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved enhanced model to {model_file}")
    
    # Create and save scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Enhanced LightGBM: Actual vs Predicted Difficulty')
    scatter_file = 'prediction_scatter_enhanced.png'
    plt.savefig(scatter_file)
    plt.close()
    logger.info(f"Saved prediction scatter plot to {scatter_file}")
    
    # Analyze feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_file = 'feature_importance_enhanced.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"Saved feature importance to {importance_file}")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features (Enhanced Model)')
    plt.tight_layout()
    importance_plot_file = 'feature_importance_enhanced.png'
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
    error_file = 'error_distribution_enhanced.png'
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
    error_analysis_file = 'error_analysis_enhanced.png'
    plt.savefig(error_analysis_file)
    plt.close()
    logger.info(f"Saved error analysis to {error_analysis_file}")
    
    return model, test_rmse, test_mae, test_r2, feature_names

def update_metrics_comparison(test_rmse, test_mae, test_r2):
    """
    Update the metrics comparison file with results from the enhanced model.
    
    Args:
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
    """
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
        
        # Create new entry for enhanced model
        new_entry = (
            f"\n### LightGBM Model with Enhanced Feature Engineering (No Count Features)\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < {DIFFICULTY_THRESHOLD} and does not use response count features\n"
            f"- Note: Uses advanced text feature extraction, TF-IDF features, clustering, and stratified sampling\n"
        )
        
        # Find the position to insert the new entry (under LightGBM Models section)
        lgbm_section = content.find("## LightGBM Models")
        next_section = content.find("##", lgbm_section + 1)
        
        if lgbm_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
            
            logger.info("Updated metrics_comparison.txt with enhanced model results.")
        else:
            logger.error("Could not update metrics_comparison.txt. Section markers not found.")
    
    except Exception as e:
        logger.error(f"Error updating metrics comparison: {str(e)}")

def main():
    """Main function to run the enhanced hyperparameter optimization"""
    start_time = time.time()
    
    # Load a sample of the data for hyperparameter optimization
    sample_df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD, sample_fraction=SAMPLE_FRACTION)
    
    # Extract features and target from sample with enhanced feature engineering
    X_sample, y_sample, feature_names_sample = preprocess_features(
        sample_df, 
        use_advanced_features=True,
        reduce_embeddings=True
    )
    
    # Create the Optuna study
    study_name = "lightgbm_enhanced"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # Optimize hyperparameters with progress tracking
    logger.info(f"Starting enhanced hyperparameter optimization with {N_TRIALS} trials...")
    
    # Define a callback to print progress after each trial
    def print_progress(study, trial):
        completed = len(study.trials)
        total = N_TRIALS
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / completed if completed > 0 else 0
        estimated_time = avg_time * (total - completed)
        logger.info(f"Progress: {completed}/{total} trials completed ({100*completed/total:.1f}%)")
        logger.info(f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_time:.1f}s")
        if completed > 0:
            logger.info(f"Current best value: {study.best_value:.4f} (trial {study.best_trial.number})")
            logger.info("--------------------")
    
    study.optimize(
        partial(objective, X=X_sample, y=y_sample, feature_names=feature_names_sample),
        n_trials=N_TRIALS,
        n_jobs=1,  # Set to 1 to see sequential progress logs
        callbacks=[print_progress],
    )
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best RMSE on sample: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save best hyperparameters
    with open('enhanced_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Now load the full dataset and train the final model
    logger.info("Now loading full dataset for final model training...")
    full_df = load_data(difficulty_threshold=DIFFICULTY_THRESHOLD, sample_fraction=1.0)
    
    # Train the final model on the full dataset with the best hyperparameters
    model, test_rmse, test_mae, test_r2, _ = train_full_model(
        best_params, full_df
    )
    
    # Update metrics comparison file
    update_metrics_comparison(test_rmse, test_mae, test_r2)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    logger.info(f"Enhanced optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final model Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    
    # Check if we beat BERT
    bert_r2 = 0.5300
    if test_r2 > bert_r2:
        logger.info(f"SUCCESS! The enhanced LightGBM model (R² = {test_r2:.4f}) outperforms BERT (R² = {bert_r2:.4f})!")
    else:
        logger.info(f"The enhanced LightGBM model (R² = {test_r2:.4f}) did not outperform BERT (R² = {bert_r2:.4f}).")
        logger.info("Consider ensemble methods to combine LightGBM and BERT predictions.")

if __name__ == "__main__":
    main() 
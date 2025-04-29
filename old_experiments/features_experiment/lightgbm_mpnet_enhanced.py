#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced LightGBM Model for Question Difficulty Prediction
This script implements an advanced LightGBM model that uses:
1. MPNet embeddings for better semantic understanding
2. Rich text features extracted from questions
3. Option similarity metrics
4. Additional categorical and numerical features

The model is designed to predict question difficulty for new questions
where response count data is not available.

This version uses the exact same features as neural_mpnet_enhanced.py
to provide a direct comparison between neural network and LightGBM approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import lightgbm as lgb
import nltk
from nltk.corpus import stopwords
import pickle
import json
import ast  # For safely evaluating the skills string representation
import re
import time
import warnings
from scipy.spatial.distance import cosine, euclidean
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DIFFICULTY_THRESHOLD = -6
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5
TFIDF_MAX_FEATURES = 50
# LightGBM specific parameters
NUM_ITERATIONS = 1000
EARLY_STOPPING_ROUNDS = 50

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

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
            'mathematical_symbols': 0,
            'latex_expressions': 0
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
    
    # Count mathematical symbols (including common symbols and operators)
    math_symbols = set(['+', '-', '*', '/', '=', '<', '>', '±', '≤', '≥', '≠', '≈', '∞', '∫', '∑', '∏', '√', '^', '÷', '×', '∆', '∇', '∂'])
    mathematical_symbols = sum(text.count(sym) for sym in math_symbols)
    
    # Count LaTeX expressions (approximate using pattern matching)
    latex_patterns = [r'\\\w+', r'\$.*?\$', r'\\\(.*?\\\)', r'\\\[.*?\\\]']
    latex_expressions = sum(len(re.findall(pattern, text)) for pattern in latex_patterns)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'mathematical_symbols': mathematical_symbols,
        'latex_expressions': latex_expressions
    }

def calculate_option_similarities(row, embedder=None):
    """
    Calculate similarities between question options.
    
    Args:
        row: DataFrame row containing option_a through option_e
        embedder: Sentence embedding model for text similarity
        
    Returns:
        Dictionary of similarity metrics
    """
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    options = [str(row[col]) if pd.notna(row[col]) else "" for col in option_cols]
    valid_options = [opt for opt in options if opt]
    
    # If less than 2 valid options, return zeros
    if len(valid_options) < 2:
        return {
            'avg_text_similarity': 0,
            'max_text_similarity': 0,
            'min_text_similarity': 0,
            'similarity_std': 0
        }
    
    # Calculate text-based similarities
    similarities = []
    
    # If embedder is provided, use semantic similarity
    if embedder:
        try:
            option_embeddings = embedder.encode(valid_options)
            
            # Calculate pairwise similarities
            for i in range(len(valid_options)):
                for j in range(i+1, len(valid_options)):
                    sim = 1 - cosine(option_embeddings[i], option_embeddings[j])
                    similarities.append(sim)
        except:
            # Fall back to length-based similarity if embedding fails
            for i in range(len(valid_options)):
                for j in range(i+1, len(valid_options)):
                    opt1, opt2 = valid_options[i], valid_options[j]
                    len_sim = 1 - abs(len(opt1) - len(opt2)) / max(len(opt1) + len(opt2), 1)
                    similarities.append(len_sim)
    
    # If no embedder or as fallback, use length-based similarity
    else:
        for i in range(len(valid_options)):
            for j in range(i+1, len(valid_options)):
                opt1, opt2 = valid_options[i], valid_options[j]
                len_sim = 1 - abs(len(opt1) - len(opt2)) / max(len(opt1) + len(opt2), 1)
                similarities.append(len_sim)
    
    # Calculate statistics from similarities
    avg_similarity = np.mean(similarities) if similarities else 0
    max_similarity = np.max(similarities) if similarities else 0
    min_similarity = np.min(similarities) if similarities else 0
    std_similarity = np.std(similarities) if len(similarities) > 1 else 0
    
    return {
        'avg_text_similarity': avg_similarity,
        'max_text_similarity': max_similarity,
        'min_text_similarity': min_similarity,
        'similarity_std': std_similarity
    }

def load_data(difficulty_threshold=DIFFICULTY_THRESHOLD):
    """
    Load and merge the question data and MPNet embeddings, filtering out very easy questions.
    
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
    
    # Load MPNet embeddings
    with open('questions_mpnet_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge dataframes on question_id
    merged_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    logger.info(f"Loaded {len(merged_df)} questions with MPNet embeddings (after filtering)")
    
    return merged_df

def preprocess_features(df):
    """
    Preprocess features with advanced text analysis and similarity metrics.
    
    Args:
        df: DataFrame with questions and embeddings
        
    Returns:
        Processed features, embeddings, target variable, and feature names
    """
    logger.info("Preprocessing features with advanced engineering...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Extract MPNet embeddings
    embeddings = np.array(df_processed['embedding'].tolist())
    logger.info(f"MPNet embedding dimension: {embeddings.shape[1]}")
    
    # Process skills
    logger.info("Processing skills...")
    mlb = MultiLabelBinarizer()
    skills_lists = df_processed['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    skills_binary = mlb.fit_transform(skills_lists)
    skills_df = pd.DataFrame(skills_binary, columns=mlb.classes_)
    
    # Convert all column names to strings
    skills_df.columns = skills_df.columns.astype(str)
    
    # Basic features without ID features (as requested)
    logger.info("Extracting basic features...")
    basic_features = ['level', 'num_misconceptions', 'has_image', 'avg_steps']
    
    # Note: Excluding topic_id, subject_id, axis_id as requested
    
    # Extract text features from question title
    logger.info("Extracting text features...")
    text_features = df_processed['question_title'].apply(
        lambda x: extract_text_features(x) if pd.notna(x) else extract_text_features("")
    )
    
    # Convert text features to DataFrame
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Create interaction features
    logger.info("Creating interaction features...")
    df_processed['math_to_text_ratio'] = text_features_df['mathematical_symbols'] / (text_features_df['word_count'] + 1)
    df_processed['digit_to_text_ratio'] = text_features_df['digit_count'] / (text_features_df['word_count'] + 1)
    df_processed['special_to_text_ratio'] = text_features_df['special_char_count'] / (text_features_df['word_count'] + 1)
    df_processed['latex_to_text_ratio'] = text_features_df['latex_expressions'] / (text_features_df['word_count'] + 1)
    
    # Question text length and complexity
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
    
    # Calculate similarity between options
    logger.info("Calculating option similarities...")
    # Initialize a lightweight sentence embedder for similarity calculations
    option_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Calculate option similarities in batches to improve performance
    similarities = []
    batch_size = 500
    for i in range(0, len(df_processed), batch_size):
        batch = df_processed.iloc[i:i+batch_size]
        batch_similarities = batch.apply(lambda row: calculate_option_similarities(row, option_embedder), axis=1)
        similarities.extend(batch_similarities)
    
    # Convert similarities to DataFrame
    similarities_df = pd.DataFrame(similarities)
    
    # Generate TF-IDF features from question text
    logger.info("Generating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, 
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
    
    # List of all features to include
    text_feature_cols = [
        'word_count', 'char_count', 'avg_word_length',
        'digit_count', 'special_char_count', 'mathematical_symbols',
        'latex_expressions', 'math_to_text_ratio', 'digit_to_text_ratio',
        'special_to_text_ratio', 'latex_to_text_ratio',
        'question_length', 'avg_option_length', 'num_options'
    ]
    
    # Combine all text features into processed dataframe
    for col in text_features_df.columns:
        df_processed[col] = text_features_df[col]
    
    # Combine features
    logger.info("Combining all features...")
    feature_dfs = [
        df_processed[basic_features + text_feature_cols].reset_index(drop=True),
        skills_df.reset_index(drop=True),
        similarities_df.reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ]
    
    X_without_embeddings = pd.concat(feature_dfs, axis=1)
    
    # Scale features
    logger.info("Scaling features...")
    feature_names = X_without_embeddings.columns.tolist()
    
    # Use Yeo-Johnson power transformation for better normalization
    scaler = PowerTransformer(method='yeo-johnson')
    X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings.fillna(0))
    
    # Combine with embeddings
    # Unlike the neural network which processes embeddings separately,
    # for LightGBM we'll include the embeddings directly in the feature matrix
    logger.info("Applying PCA to reduce embedding dimensions...")
    # Apply PCA to reduce dimensionality of embeddings
    pca = PCA(n_components=50)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    # Create final feature matrix including embeddings
    X_with_embeddings = np.hstack([X_without_embeddings_scaled, embeddings_reduced])
    
    # Update feature names to include embedding dimensions
    embedding_feature_names = [f'embedding_pca_{i}' for i in range(embeddings_reduced.shape[1])]
    all_feature_names = feature_names + embedding_feature_names
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
        'scaler': scaler,
        'tfidf': tfidf,
        'pca': pca
    }
    
    with open('feature_processors_lightgbm_mpnet_enhanced.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    logger.info(f"Final feature matrix shape: {X_with_embeddings.shape}")
    logger.info(f"Number of feature names: {len(all_feature_names)}")
    
    return X_with_embeddings, y, all_feature_names

def train_and_evaluate_model(X, y, feature_names):
    """
    Train and evaluate a LightGBM model with cross-validation.
    
    Args:
        X (numpy.ndarray): Features matrix
        y (numpy.ndarray): Target values
        feature_names (list): Names of features
        
    Returns:
        tuple: Mean RMSE, MAE, and R² from cross-validation
    """
    logging.info("Starting 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
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
    
    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        logging.info(f"Training fold {fold_idx}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=callbacks,
            num_boost_round=1000
        )
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        logging.info(f"Fold {fold_idx} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        fold_idx += 1
    
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)
    mean_r2 = np.mean(r2_scores)
    
    logging.info(f"Cross-validation - Avg RMSE: {mean_rmse:.4f}, Avg MAE: {mean_mae:.4f}, Avg R²: {mean_r2:.4f}")
    
    # Train model on full dataset
    logging.info("Training final model on full dataset...")
    
    # Split data into train and test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100, show_stdv=False)
    ]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        callbacks=callbacks,
        num_boost_round=1000
    )
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics on test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Test set - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Log feature importance
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Log top 10 features
    logging.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logging.info(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Save feature importance to CSV
    feature_importance_file = "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_file, index=False)
    logging.info(f"Feature importance saved to {feature_importance_file}")
    
    # Update metrics comparison
    update_metrics_comparison("LightGBM_MPNet_Enhanced", test_rmse, test_mae, test_r2)
    
    return mean_rmse, mean_mae, mean_r2

def update_metrics_comparison(model_name, rmse, mae, r2):
    """
    Update the metrics comparison file with the results of the current model.
    
    Args:
        model_name: Name of the model
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R-squared
    """
    comparison_file = os.path.join(os.path.dirname(__file__), "model_comparison.csv")
    
    # Create a new DataFrame for this model's metrics
    new_data = pd.DataFrame({
        'model': [model_name],
        'rmse': [rmse],
        'mae': [mae],
        'r2': [r2],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    
    # If the file exists, append to it; otherwise, create it
    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
        comparison_df = pd.concat([comparison_df, new_data], ignore_index=True)
    else:
        comparison_df = new_data
    
    # Save the updated comparison
    comparison_df.to_csv(comparison_file, index=False)
    logging.info(f"Updated metrics comparison in {comparison_file}")

def main():
    """Main function to run the enhanced LightGBM training pipeline."""
    start_time = time.time()
    
    # Load data
    df = load_data()
    
    # Preprocess features
    X, y, feature_names = preprocess_features(df)
    
    # Train and evaluate model
    mean_rmse, mean_mae, mean_r2 = train_and_evaluate_model(X, y, feature_names)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main() 
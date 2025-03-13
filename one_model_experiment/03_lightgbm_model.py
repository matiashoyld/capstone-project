import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def load_data():
    """
    Load the processed data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df

def load_embeddings():
    """
    Load the option embeddings and question embeddings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    option_embeddings_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    question_embeddings_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    
    embeddings = {}
    
    # Load option embeddings
    if os.path.exists(option_embeddings_path):
        print(f"Loading option embeddings from {option_embeddings_path}")
        with open(option_embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print(f"Warning: Option embeddings file {option_embeddings_path} not found")
        embeddings = None
    
    # Load question embeddings
    if os.path.exists(question_embeddings_path):
        print(f"Loading question embeddings from {question_embeddings_path}")
        with open(question_embeddings_path, 'rb') as f:
            question_embeddings = pickle.load(f)
        
        # Add question embeddings to the embeddings dictionary
        if embeddings is not None:
            # The formatted_embeddings key contains the question embeddings
            if 'formatted_embeddings' in question_embeddings:
                embeddings['question_embeddings'] = question_embeddings['formatted_embeddings']
                print("Question embeddings found under 'formatted_embeddings' key")
            elif 'question_embeddings' in question_embeddings:
                embeddings['question_embeddings'] = question_embeddings['question_embeddings']
                print("Question embeddings found under 'question_embeddings' key")
            else:
                print("Warning: No question embeddings found in the file")
        else:
            if 'formatted_embeddings' in question_embeddings:
                embeddings = {'question_embeddings': question_embeddings['formatted_embeddings']}
                print("Question embeddings found under 'formatted_embeddings' key")
            elif 'question_embeddings' in question_embeddings:
                embeddings = {'question_embeddings': question_embeddings['question_embeddings']}
                print("Question embeddings found under 'question_embeddings' key")
            else:
                print("Warning: No question embeddings found in the file")
    else:
        print(f"Warning: Question embeddings file {question_embeddings_path} not found")
    
    return embeddings

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_option_similarities(question_id, correct_option_letter, embeddings):
    """
    Calculate cosine similarities between the correct option and each wrong option.
    
    Args:
        question_id: Question ID
        correct_option_letter: Letter of the correct option
        embeddings: Dictionary of embeddings
        
    Returns:
        Dictionary mapping wrong option letters to similarities with correct option
    """
    # Map option letters to their indices and column names
    option_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    
    # Check if correct_option_letter is valid
    if correct_option_letter not in option_indices:
        return {f'wrong_{i+1}_similarity': 0 for i in range(4)}
    
    # Find index of question_id in embeddings list
    try:
        q_idx = embeddings['question_ids'].index(question_id)
    except ValueError:
        # If question_id not found in embeddings, return zeros
        return {f'wrong_{i+1}_similarity': 0 for i in range(4)}
    
    # Get index and column name for correct option
    correct_idx = option_indices[correct_option_letter]
    correct_col = option_cols[correct_idx]
    
    # Check if correct option embedding exists
    if correct_col not in embeddings['option_embeddings'] or q_idx >= len(embeddings['option_embeddings'][correct_col]):
        return {f'wrong_{i+1}_similarity': 0 for i in range(4)}
    
    # Get embedding for correct option
    try:
        correct_embedding = embeddings['option_embeddings'][correct_col][q_idx]
    except IndexError:
        # Handle index error gracefully
        return {f'wrong_{i+1}_similarity': 0 for i in range(4)}
    
    # Calculate similarities with wrong options
    similarities = {}
    wrong_indices = [i for i in range(5) if i != correct_idx]
    
    for i, wrong_idx in enumerate(wrong_indices[:4]):  # Limit to 4 wrong options
        wrong_col = option_cols[wrong_idx]
        
        try:
            # Check if wrong option embedding exists
            if wrong_col not in embeddings['option_embeddings'] or q_idx >= len(embeddings['option_embeddings'][wrong_col]):
                similarities[f'wrong_{i+1}_similarity'] = 0
                continue
                
            wrong_embedding = embeddings['option_embeddings'][wrong_col][q_idx]
            
            # Calculate cosine similarity
            sim = cosine_similarity(correct_embedding, wrong_embedding)
            similarities[f'wrong_{i+1}_similarity'] = sim
        except Exception as e:
            # Print detailed error message for debugging
            print(f"Error calculating similarity for question {question_id}, option {wrong_col}: {e}")
            similarities[f'wrong_{i+1}_similarity'] = 0
    
    # Make sure we have exactly 4 similarity values
    for i in range(4):
        if f'wrong_{i+1}_similarity' not in similarities:
            similarities[f'wrong_{i+1}_similarity'] = 0
    
    return similarities

def stratified_question_split(df, test_size=0.2, random_state=42):
    """
    Split questions into train and test sets in a stratified manner based on 
    correct answer ratios to ensure balanced distributions.
    
    Returns question_ids for train and test sets.
    """
    # Get unique questions and their metrics
    question_df = df.groupby('question_id').agg({
        'is_correct': 'mean'  # Average correctness rate per question
    }).reset_index()
    
    # Bin correctness rate into 5 bins
    question_df['correctness_bin'] = pd.qcut(question_df['is_correct'], 5, labels=False)
    
    # Split the questions using stratification
    train_questions, test_questions = train_test_split(
        question_df,
        test_size=test_size,
        random_state=random_state,
        stratify=question_df['correctness_bin']
    )
    
    print(f"Train set: {len(train_questions)} questions")
    print(f"Test set: {len(test_questions)} questions")
    
    # Verify distributions
    print("\nCorrectness distribution:")
    print("Train mean: {:.4f}, std: {:.4f}".format(
        train_questions['is_correct'].mean(), 
        train_questions['is_correct'].std()
    ))
    print("Test mean: {:.4f}, std: {:.4f}".format(
        test_questions['is_correct'].mean(), 
        test_questions['is_correct'].std()
    ))
    
    return train_questions['question_id'].tolist(), test_questions['question_id'].tolist()

def prepare_features(df, train_question_ids, test_question_ids, embeddings=None):
    """
    Prepare features for model training and testing excluding specified features
    and adding cosine similarity features and one-hot user_id.
    """
    # Split data based on question IDs
    train_df = df[df['question_id'].isin(train_question_ids)]
    test_df = df[df['question_id'].isin(test_question_ids)]
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    # Define feature columns to keep
    feature_columns = [
        # Text features
        'title_word_count', 'title_char_count', 'title_avg_word_length',
        'title_digit_count', 'title_special_char_count', 
        'title_mathematical_symbols', 'title_latex_expressions',
        
        # Answer option features
        'jaccard_similarity_std', 'avg_option_length', 'avg_option_word_count',
        
        # Has image feature
        'has_image',
        
        # Question metadata
        'avg_steps', 'level', 'num_misconceptions'
    ]
    
    # Create initial train and test feature sets
    train_X = train_df[feature_columns].copy()
    test_X = test_df[feature_columns].copy()
    
    # Fill missing values appropriately for each column
    for col in train_X.columns:
        if train_X[col].dtype == np.float64 or train_X[col].dtype == np.int64:
            # For numeric columns, use median
            fill_value = train_X[col].median()
            train_X[col] = train_X[col].fillna(fill_value)
            test_X[col] = test_X[col].fillna(fill_value)
        else:
            # For categorical columns, use mode (most frequent value)
            fill_value = train_X[col].mode()[0]
            train_X[col] = train_X[col].fillna(fill_value)
            test_X[col] = test_X[col].fillna(fill_value)
    
    # Convert boolean to int
    for col in train_X.columns:
        if train_X[col].dtype == np.bool_:
            train_X[col] = train_X[col].astype(int)
            test_X[col] = test_X[col].astype(int)
    
    # Prepare targets
    train_y = train_df['is_correct'].astype(int)
    test_y = test_df['is_correct'].astype(int)
    
    # Add PCA-reduced question embeddings (if available)
    if embeddings is not None and 'question_embeddings' in embeddings:
        print("Adding PCA-reduced question embeddings...")
        
        # Create a mapping from question_id to index in embeddings
        question_id_to_idx = {}
        if 'question_ids' in embeddings:
            question_id_to_idx = {qid: idx for idx, qid in enumerate(embeddings['question_ids'])}
        
        # Create a DataFrame to hold question embeddings for all questions in train and test
        all_question_ids = set(train_question_ids).union(set(test_question_ids))
        question_embeddings_array = []
        question_ids_with_embeddings = []
        
        print(f"Question embeddings shape: {embeddings['question_embeddings'].shape if hasattr(embeddings['question_embeddings'], 'shape') else 'unknown'}")
        
        for qid in all_question_ids:
            if qid in question_id_to_idx:
                idx = question_id_to_idx[qid]
                if idx < len(embeddings['question_embeddings']):
                    question_embeddings_array.append(embeddings['question_embeddings'][idx])
                    question_ids_with_embeddings.append(qid)
        
        if question_embeddings_array:
            # Convert to numpy array
            question_embeddings_array = np.array(question_embeddings_array)
            print(f"Question embeddings array shape: {question_embeddings_array.shape}")
            
            # Apply PCA to reduce dimensionality to 50
            pca = PCA(n_components=50)
            reduced_embeddings = pca.fit_transform(question_embeddings_array)
            print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
            print(f"Explained variance ratio sum: {sum(pca.explained_variance_ratio_):.4f}")
            
            # Create a mapping from question_id to PCA-reduced embeddings
            question_id_to_embedding = {qid: emb for qid, emb in zip(question_ids_with_embeddings, reduced_embeddings)}
            
            # Add embeddings to train_X
            for i in range(50):
                train_X[f'q_emb_{i}'] = 0.0
                test_X[f'q_emb_{i}'] = 0.0
            
            # Fill in embeddings for each question in train_X
            for idx, row in train_df.iterrows():
                qid = row['question_id']
                if qid in question_id_to_embedding:
                    for i in range(50):
                        train_X.at[idx, f'q_emb_{i}'] = question_id_to_embedding[qid][i]
            
            # Fill in embeddings for each question in test_X
            for idx, row in test_df.iterrows():
                qid = row['question_id']
                if qid in question_id_to_embedding:
                    for i in range(50):
                        test_X.at[idx, f'q_emb_{i}'] = question_id_to_embedding[qid][i]
            
            print(f"Added 50 PCA-reduced question embedding features")
        else:
            print("Warning: No question embeddings found for any questions in the dataset!")
    
    # One-hot encode skills
    print("One-hot encoding skills...")
    
    # Process skills column - convert string representation of list to actual list
    def process_skills(skills_str):
        if pd.isna(skills_str) or skills_str == '[]':
            return []
        try:
            if isinstance(skills_str, str):
                # Remove brackets and split by comma
                skills_str = skills_str.strip('[]').replace(' ', '')
                if skills_str:
                    return [int(skill) for skill in skills_str.split(',')]
                else:
                    return []
            elif isinstance(skills_str, list):
                return skills_str
            else:
                return []
        except Exception as e:
            print(f"Error processing skills {skills_str}: {e}")
            return []
    
    # Apply skills processing
    train_df['skills_list'] = train_df['skills'].apply(process_skills)
    test_df['skills_list'] = test_df['skills'].apply(process_skills)
    
    # Find all unique skills across train and test sets
    all_skills = set()
    for skills in train_df['skills_list'].values:
        all_skills.update(skills)
    for skills in test_df['skills_list'].values:
        all_skills.update(skills)
    
    print(f"Found {len(all_skills)} unique skills")
    
    # Create one-hot encoding for skills
    for skill_id in all_skills:
        skill_col = f'skill_{skill_id}'
        # For train set
        train_X[skill_col] = train_df['skills_list'].apply(lambda x: 1 if skill_id in x else 0)
        # For test set
        test_X[skill_col] = test_df['skills_list'].apply(lambda x: 1 if skill_id in x else 0)
    
    print(f"Added {len(all_skills)} skill one-hot features")
    
    # Add cosine similarity features if embeddings are available
    if embeddings is not None:
        print("Adding cosine similarity features between options...")
        
        # Precompute a dictionary of similarities for all question_ids in the embeddings
        # This avoids repeated lookups and calculations
        print("Precomputing similarities for all questions...")
        
        # Get the mapping from the dataset
        unique_questions = pd.concat([train_df, test_df])[['question_id', 'correct_option_letter']].drop_duplicates()
        
        # Convert correct_option_letter to uppercase
        unique_questions['correct_option_letter'] = unique_questions['correct_option_letter'].str.upper()
        
        question_to_correct = dict(zip(unique_questions['question_id'], unique_questions['correct_option_letter']))
        
        # Debug: Print the first few mappings
        print(f"Total unique questions in dataset: {len(question_to_correct)}")
        print(f"Sample of question_id to correct_option_letter mapping (after uppercase conversion):")
        print(list(question_to_correct.items())[:5])
        
        # Map option letters to option columns
        option_map = {'A': 'option_a', 'B': 'option_b', 'C': 'option_c', 'D': 'option_d', 'E': 'option_e'}
        
        # Debug: Print some question_ids from embeddings
        print(f"Total question_ids in embeddings: {len(embeddings['question_ids'])}")
        print(f"Sample of question_ids in embeddings: {embeddings['question_ids'][:5]}")
        
        # Check for overlap between dataset and embeddings
        common_questions = set(question_to_correct.keys()).intersection(set(embeddings['question_ids']))
        print(f"Number of questions common to both dataset and embeddings: {len(common_questions)}")
        
        # Dictionary to store similarities for each question
        similarities_dict = {}
        
        # For each question in the embeddings
        debug_count = 0
        for i, question_id in enumerate(tqdm(embeddings['question_ids'], desc="Computing similarities")):
            # Check if this question is in our dataset
            if question_id not in question_to_correct:
                continue
                
            # Debug counter
            debug_count += 1
            if debug_count <= 5:
                print(f"Processing question {question_id}")
                
            correct_letter = question_to_correct[question_id]
            if correct_letter not in option_map:
                if debug_count <= 5:
                    print(f"  Invalid correct letter: {correct_letter}")
                continue
                
            # Get the correct option column
            correct_col = option_map[correct_letter]
            
            # Check if the correct option has an embedding
            if correct_col not in embeddings['option_embeddings'] or i >= len(embeddings['option_embeddings'][correct_col]):
                if debug_count <= 5:
                    print(f"  Missing embedding for {correct_col} at index {i}")
                continue
                
            # Get the embedding for the correct option
            try:
                correct_embedding = embeddings['option_embeddings'][correct_col][i]
                if debug_count <= 5:
                    print(f"  Got embedding for {correct_col} with shape {correct_embedding.shape}")
            except Exception as e:
                if debug_count <= 5:
                    print(f"  Error getting embedding for {correct_col}: {e}")
                continue
                
            # Calculate similarities with wrong options
            wrong_cols = [option_map[letter] for letter in option_map if letter != correct_letter]
            
            question_similarities = {}
            for j, wrong_col in enumerate(wrong_cols[:4]):  # Limit to 4 wrong options
                if wrong_col not in embeddings['option_embeddings'] or i >= len(embeddings['option_embeddings'][wrong_col]):
                    question_similarities[f'wrong_{j+1}_similarity'] = 0
                    if debug_count <= 5:
                        print(f"  Missing embedding for {wrong_col}")
                    continue
                    
                # Get embedding for wrong option
                try:
                    wrong_embedding = embeddings['option_embeddings'][wrong_col][i]
                    # Calculate cosine similarity
                    sim = cosine_similarity(correct_embedding, wrong_embedding)
                    question_similarities[f'wrong_{j+1}_similarity'] = sim
                    if debug_count <= 5 and j < 2:  # Just show first 2 to avoid too much output
                        print(f"  Similarity between {correct_col} and {wrong_col}: {sim:.4f}")
                except Exception as e:
                    if debug_count <= 5:
                        print(f"  Error calculating similarity for {wrong_col}: {e}")
                    question_similarities[f'wrong_{j+1}_similarity'] = 0
            
            # Make sure we have exactly 4 similarity values
            for j in range(4):
                if f'wrong_{j+1}_similarity' not in question_similarities:
                    question_similarities[f'wrong_{j+1}_similarity'] = 0
                    
            # Store similarities for this question
            similarities_dict[question_id] = question_similarities
        
        print(f"Computed similarities for {len(similarities_dict)} questions")
        
        # Add similarities to train_X and test_X
        # First create empty columns for wrong_1-4_similarity
        for j in range(4):
            train_X[f'wrong_{j+1}_similarity'] = 0.0
            test_X[f'wrong_{j+1}_similarity'] = 0.0
        
        # If we found any similarities, show a sample
        if similarities_dict:
            sample_question_id = next(iter(similarities_dict))
            print(f"Sample similarities for question {sample_question_id}: {similarities_dict[sample_question_id]}")
            
            # Fill in the similarities for each question in train_X
            for idx, row in train_df.iterrows():
                question_id = row['question_id']
                if question_id in similarities_dict:
                    for key, value in similarities_dict[question_id].items():
                        train_X.at[idx, key] = value
            
            # Fill in the similarities for each question in test_X
            for idx, row in test_df.iterrows():
                question_id = row['question_id']
                if question_id in similarities_dict:
                    for key, value in similarities_dict[question_id].items():
                        test_X.at[idx, key] = value
        else:
            print("WARNING: No similarities computed! Proceeding with zero values for similarity features.")
    
    # Add one-hot encoding for user_id (using get_dummies for better indexing)
    print("One-hot encoding user_id...")
    
    # Create one-hot encoding for user_id
    # First, create a combined DataFrame with a 'in_train' column to track source
    train_df_tmp = train_df[['user_id']].copy()
    train_df_tmp['in_train'] = True
    test_df_tmp = test_df[['user_id']].copy()
    test_df_tmp['in_train'] = False
    
    # Combine both datasets for encoding
    combined_df = pd.concat([train_df_tmp, test_df_tmp])
    
    # Create one-hot encoding
    user_id_dummies = pd.get_dummies(combined_df['user_id'], prefix='user_id', sparse=True)
    
    # Add the 'in_train' column back
    user_id_dummies = pd.concat([user_id_dummies, combined_df['in_train']], axis=1)
    
    # Split back into train and test
    train_user_id = user_id_dummies[user_id_dummies['in_train']].drop('in_train', axis=1)
    test_user_id = user_id_dummies[~user_id_dummies['in_train']].drop('in_train', axis=1)
    
    # Ensure train_user_id has the same index as train_df
    train_user_id.index = train_df.index
    test_user_id.index = test_df.index
    
    # Add one-hot encoded user_id
    train_X = pd.concat([train_X, train_user_id], axis=1)
    test_X = pd.concat([test_X, test_user_id], axis=1)
    
    # Print final train_X columns (non-user_id only to avoid clutter)
    non_user_cols = [col for col in train_X.columns if not col.startswith('user_id_')]
    print(f"\nFinal feature columns (excluding user_id features): {non_user_cols}")
    
    # Standardize numeric features (except one-hot encoded features)
    numeric_cols = [col for col in train_X.columns 
                   if not col.startswith('user_id_') 
                   and not col.startswith('skill_')]
    scaler = StandardScaler()
    
    train_X[numeric_cols] = scaler.fit_transform(train_X[numeric_cols])
    test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])
    
    return train_X, train_y, test_X, test_y, train_df, test_df

def train_and_evaluate_model(train_X, train_y, test_X, test_y):
    """
    Train a LightGBM model and evaluate its performance.
    """
    # Create a LightGBM classifier
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Convert data to LightGBM dataset format
    train_data = lgb.Dataset(train_X, label=train_y)
    valid_data = lgb.Dataset(test_X, label=test_y, reference=train_data)
    
    # Train the model
    print("\nTraining LightGBM model...")
    model = lgb.train(
        params=lgb_params, 
        train_set=train_data, 
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Make predictions
    test_preds_proba = model.predict(test_X)
    test_preds = (test_preds_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(test_y, test_preds)
    precision = precision_score(test_y, test_preds)
    recall = recall_score(test_y, test_preds)
    f1 = f1_score(test_y, test_preds)
    auc = roc_auc_score(test_y, test_preds_proba)
    
    print("\nModel performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(test_y, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_y, test_preds))
    
    # Get feature importance
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = train_X.columns
    
    # Filter out user_id features for visualization (too many)
    non_user_features = [col for col in feature_names if not col.startswith('user_id_')]
    non_user_importance = [imp for col, imp in zip(feature_names, feature_importance) if not col.startswith('user_id_')]
    
    # Create DataFrame for non-user features for visualization
    importance_df_viz = pd.DataFrame({
        'Feature': non_user_features,
        'Importance': non_user_importance
    }).sort_values(by='Importance', ascending=False)
    
    # Create complete DataFrame for all features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    # Save full feature importance to CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    importance_csv_path = os.path.join(current_dir + '/features/03_lightgbm_model.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"\nFeature importance saved to {importance_csv_path}")
    
    print("\nTop 20 Important Features (excluding user_id):")
    print(importance_df_viz.head(20))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df_viz.head(20))
    plt.title('Feature Importance (Gain) - Excluding user_id')
    plt.tight_layout()
    
    plt.savefig(os.path.join(current_dir + '/images/03_lightgbm_model.png'))
    
    # Save model
    model_path = os.path.join(current_dir + '/models/03_lightgbm_model.txt')
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, importance_df

def main():
    # Load data
    df = load_data()
    
    # Load embeddings
    embeddings = load_embeddings()
    if embeddings is None:
        print("Error: Embeddings are required for this refined model.")
        return
    
    # Split questions
    train_question_ids, test_question_ids = stratified_question_split(df)
    
    # Prepare features
    train_X, train_y, test_X, test_y, train_df, test_df = prepare_features(
        df, train_question_ids, test_question_ids, embeddings
    )
    
    # Train and evaluate model
    model, importance_df = train_and_evaluate_model(train_X, train_y, test_X, test_y)
    
    # Save results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = {
        'train_shape': train_X.shape,
        'test_shape': test_X.shape,
        'feature_importance': importance_df.to_dict(),
        'train_question_ids': train_question_ids,
        'test_question_ids': test_question_ids
    }
    
    with open(os.path.join(current_dir + '/models/03_lightgbm_model.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 
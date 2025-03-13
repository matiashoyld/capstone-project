import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_trained_model():
    """
    Load the trained LightGBM model.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'lightgbm_model_refined.txt')
    print(f"Loading model from {model_path}")
    model = lgb.Booster(model_file=model_path)
    return model

def load_embeddings():
    """
    Load the option embeddings and question embeddings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded from {embeddings_path}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def get_top_user_ids(n=100):
    """
    Get the top n user IDs from the feature importance CSV.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    importance_path = os.path.join(current_dir, 'features', '03_lightgbm_model.csv')
    
    # Read the feature importance CSV
    importance_df = pd.read_csv(importance_path)
    
    # Filter for user_id features and sort by importance
    user_features = importance_df[importance_df['Feature'].str.startswith('user_id_')]
    
    # Sort by importance descending
    user_features = user_features.sort_values('Importance', ascending=False)
    
    # Extract user IDs from feature names and convert to integers
    top_users = user_features.head(n)['Feature'].apply(lambda x: int(x.split('_')[-1])).tolist()
    
    print(f"Extracted top {len(top_users)} user IDs")
    return top_users

def get_test_question_ids(df):
    """
    Get the test question IDs used in the LightGBM model.
    """
    _, test_question_ids = stratified_question_split(df)
    print(f"Retrieved {len(test_question_ids)} test question IDs")
    return test_question_ids

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
    from sklearn.model_selection import train_test_split
    train_questions, test_questions = train_test_split(
        question_df,
        test_size=test_size,
        random_state=random_state,
        stratify=question_df['correctness_bin']
    )
    
    return train_questions['question_id'].tolist(), test_questions['question_id'].tolist()

def prepare_features(df, user_id, question_id, embeddings=None):
    """
    Prepare features for a single user-question pair, exactly matching the training process.
    """
    # Filter data for the specific question
    question_data = df[df['question_id'] == question_id].iloc[0].to_dict()
    
    # Define feature columns to keep (exactly as in the original model)
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
    
    # Create features dictionary with only the relevant columns
    features = {}
    for col in feature_columns:
        if col in question_data:
            features[col] = question_data[col]
        else:
            print(f"Warning: Column {col} not found in question data")
            features[col] = 0  # Use a default value
    
    # Process numeric and boolean values
    for col in features:
        # Handle missing values for numeric columns
        if pd.isna(features[col]) and (isinstance(features[col], (int, float)) or features[col] is None):
            features[col] = 0  # Use a default value, ideally should be the median from training
            
        # Convert boolean to int
        if isinstance(features[col], bool):
            features[col] = int(features[col])
    
    # Add PCA for embeddings if available (similar to training)
    if embeddings is not None:
        # Find index of the question in embeddings
        try:
            q_idx = embeddings['question_ids'].index(question_id)
            
            # Add embedding features
            for i in range(50):  # Use 50 dimensions as in training
                if q_idx < len(embeddings['formatted_embeddings']) and i < len(embeddings['formatted_embeddings'][q_idx]):
                    features[f'q_emb_{i}'] = embeddings['formatted_embeddings'][q_idx][i]
                else:
                    features[f'q_emb_{i}'] = 0
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not find embeddings for question {question_id}: {e}")
            # Fill with zeros if not found
            for i in range(50):
                features[f'q_emb_{i}'] = 0
    
    # Process skills for one-hot encoding (exactly as in training)
    if 'skills' in question_data:
        # Process skills string to list
        skills_str = question_data['skills']
        skills_list = []
        
        if not pd.isna(skills_str) and skills_str != '[]':
            try:
                if isinstance(skills_str, str):
                    # Remove brackets and split by comma
                    skills_str = skills_str.strip('[]').replace(' ', '')
                    if skills_str:
                        skills_list = [int(skill) for skill in skills_str.split(',')]
                elif isinstance(skills_str, list):
                    skills_list = skills_str
            except Exception as e:
                print(f"Error processing skills {skills_str}: {e}")
        
        # Set skill features to 0 by default (will be filled in by model feature processing)
        # In the prediction code, we'll set appropriate ones to 1 based on feature names
        
    return features, user_id, skills_list if 'skills_list' in locals() else []

def make_prediction_matrix_weighted(df, top_user_ids, test_question_ids, model, embeddings=None, 
                                   feature_weights={}, threshold=0.5, default_weight=10.0, user_weight=0.3):
    """
    Create a binary prediction matrix where rows are questions and columns are users.
    Uses importance-based feature weights for better balance.
    
    Args:
        df: DataFrame with raw data
        top_user_ids: List of user IDs to include in matrix
        test_question_ids: List of question IDs to include in matrix
        model: Trained LightGBM model
        embeddings: Question embeddings dictionary
        feature_weights: Dictionary mapping feature names to weights
        threshold: Threshold for binary prediction (default 0.5)
        default_weight: Default weight for features not explicitly weighted (default 10.0)
        user_weight: Weight for user ID features (default 0.3)
    """
    # Create an empty matrix
    matrix = np.zeros((len(test_question_ids), len(top_user_ids)))
    proba_matrix = np.zeros((len(test_question_ids), len(top_user_ids)))
    
    # Get the feature names expected by the model
    feature_names = model.feature_name()
    
    # Extract specific feature types
    user_id_features = [f for f in feature_names if f.startswith('user_id_')]
    skill_features = [f for f in feature_names if f.startswith('skill_')]
    q_emb_features = [f for f in feature_names if f.startswith('q_emb_')]
    other_features = [f for f in feature_names if not f.startswith('user_id_') 
                      and not f.startswith('skill_') and not f.startswith('q_emb_')]
    
    # All question-related features
    question_features = skill_features + q_emb_features + other_features
    
    print(f"Model expects {len(feature_names)} features:")
    print(f" - {len(user_id_features)} user_id features")
    print(f" - {len(skill_features)} skill features") 
    print(f" - {len(q_emb_features)} embedding features")
    print(f" - {len(other_features)} other features")
    print(f"Using importance-based feature weights with default={default_weight}x")
    
    # Debug: track feature values for specific questions
    debug_question_indices = [0, 1, 2]  # First 3 questions
    debug_user_indices = [0, 1]         # First 2 users
    
    print("Generating prediction matrix...")
    # For each question
    for i, question_id in enumerate(tqdm(test_question_ids)):
        # For each user
        for j, user_id in enumerate(top_user_ids):
            # Prepare base features (non-user features)
            features, _, skills_list = prepare_features(df, user_id, question_id, embeddings)
            
            # Convert to a format suitable for LightGBM prediction
            feature_values = []
            for feature in feature_names:
                if feature.startswith('user_id_'):
                    # If this is the current user's feature, apply user weight
                    feature_values.append(user_weight if feature == f'user_id_{user_id}' else 0)
                elif feature.startswith('skill_'):
                    # Extract skill ID from feature name
                    skill_id = int(feature.split('_')[-1])
                    # Get weight (default or from weights dict)
                    weight = feature_weights.get(feature, default_weight)
                    # Set to weight if this skill is in the question's skills list
                    feature_values.append(weight * (1 if skill_id in skills_list else 0))
                elif feature in feature_weights:
                    # Use importance-based weight for high-importance features
                    feature_values.append(feature_weights[feature] * features.get(feature, 0))
                elif feature in features:
                    # Use default weight for other features
                    feature_values.append(default_weight * features.get(feature, 0))
                else:
                    # For features not found, default to 0
                    feature_values.append(0)
            
            # Debug output for specific questions and users
            if i in debug_question_indices and j in debug_user_indices:
                print(f"\nDEBUG - Question {question_id} (idx {i}) with User {user_id} (idx {j}):")
                
                # Count non-zero feature values by category
                user_features_nonzero = sum(1 for idx, f in enumerate(feature_names) 
                                          if f.startswith('user_id_') and feature_values[idx] != 0)
                skill_features_nonzero = sum(1 for idx, f in enumerate(feature_names) 
                                           if f.startswith('skill_') and feature_values[idx] != 0)
                emb_features_nonzero = sum(1 for idx, f in enumerate(feature_names) 
                                         if f.startswith('q_emb_') and feature_values[idx] != 0)
                other_features_nonzero = sum(1 for idx, f in enumerate(feature_names) 
                                           if not (f.startswith('user_id_') or f.startswith('skill_') or f.startswith('q_emb_')) 
                                           and feature_values[idx] != 0)
                
                print(f"  Non-zero features: user={user_features_nonzero}, skill={skill_features_nonzero}, "
                      f"emb={emb_features_nonzero}, other={other_features_nonzero}")
                
                # Show some actual feature values with weights
                print("  Sample weighted feature values:")
                for category, prefix in [("User", "user_id_"), ("Skill", "skill_"), 
                                        ("Embedding", "q_emb_"), ("Other", "")]:
                    if category == "Other":
                        sample_features = [(idx, f, feature_values[idx]) 
                                          for idx, f in enumerate(feature_names) 
                                          if not (f.startswith('user_id_') or f.startswith('skill_') 
                                                or f.startswith('q_emb_'))][:5]
                    else:
                        sample_features = [(idx, f, feature_values[idx]) 
                                          for idx, f in enumerate(feature_names) 
                                          if f.startswith(prefix)][:5]
                    
                    print(f"    {category}: {sample_features}")
                    
                # Check for missing features
                missing_count = feature_values.count(0)
                print(f"  Missing/zero values: {missing_count} out of {len(feature_values)} "
                      f"({missing_count/len(feature_values)*100:.1f}%)")
            
            # Make prediction (get probability)
            prediction_proba = model.predict([feature_values])[0]
            proba_matrix[i, j] = prediction_proba
            
            # Apply threshold to convert to binary (0 or 1)
            prediction_binary = 1 if prediction_proba > threshold else 0
            
            # Store in matrix
            matrix[i, j] = prediction_binary
    
    # After generating all predictions, analyze the probabilities
    print("\nProbability Matrix Analysis:")
    print(f"Min: {proba_matrix.min():.4f}, Max: {proba_matrix.max():.4f}, Mean: {proba_matrix.mean():.4f}")
    
    # Check how much variation exists across questions for each user
    user_stds = np.std(proba_matrix, axis=0)
    print(f"Std dev across questions (per user) - Min: {user_stds.min():.6f}, Max: {user_stds.max():.6f}, Mean: {user_stds.mean():.6f}")
    
    # Check if any users have the same prediction for all questions
    identical_pred_users = np.sum(user_stds < 0.001)
    print(f"Users with identical predictions for all questions: {identical_pred_users} out of {len(top_user_ids)}")
    
    # Check how much variation exists across users for each question
    question_stds = np.std(proba_matrix, axis=1)
    print(f"Std dev across users (per question) - Min: {question_stds.min():.6f}, Max: {question_stds.max():.6f}, Mean: {question_stds.mean():.6f}")
    
    # Save both matrices for analysis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, 'predictions'), exist_ok=True)
    
    proba_df = pd.DataFrame(proba_matrix, index=test_question_ids, columns=top_user_ids)
    proba_path = os.path.join(current_dir, 'predictions', '04_probability_matrix.csv')
    proba_df.to_csv(proba_path)
    print(f"\nSaved probability matrix to {proba_path}")
    
    return matrix, proba_matrix

def save_prediction_matrix(matrix, test_question_ids, top_user_ids, proba_matrix=None):
    """
    Save the binary prediction matrix to a CSV file.
    """
    # Create a DataFrame from the matrix
    matrix_df = pd.DataFrame(
        matrix, 
        index=test_question_ids, 
        columns=top_user_ids
    )
    
    # Save to CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'predictions', '04_binary_prediction_matrix.csv')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    matrix_df.to_csv(output_path)
    print(f"Binary prediction matrix saved to {output_path}")
    
    # Generate visualizations
    # Save a heatmap visualization with binary colormap
    plt.figure(figsize=(20, 10))
    sns.heatmap(matrix, cmap='Greys', cbar=False)  # Binary colormap
    plt.title('Binary Prediction Matrix (Questions x Users, threshold=0.5)')
    plt.xlabel('User IDs (index)')
    plt.ylabel('Question IDs (index)')
    
    heatmap_path = os.path.join(current_dir, 'images', '04_binary_prediction_matrix.png')
    plt.savefig(heatmap_path)
    print(f"Binary heatmap saved to {heatmap_path}")
    
    # Also create a heatmap of the probability matrix if provided
    if proba_matrix is not None:
        plt.figure(figsize=(20, 10))
        sns.heatmap(proba_matrix, cmap='coolwarm')
        plt.title('Probability Matrix (Questions x Users)')
        plt.xlabel('User IDs (index)')
        plt.ylabel('Question IDs (index)')
        
        proba_heatmap_path = os.path.join(current_dir, 'images', '04_probability_matrix.png')
        plt.savefig(proba_heatmap_path)
        print(f"Probability heatmap saved to {proba_heatmap_path}")

def main():
    # Load data
    df = load_data()
    
    # Load model
    model = load_trained_model()
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Get top user IDs
    top_user_ids = get_top_user_ids(100)
    
    # Get test question IDs
    test_question_ids = get_test_question_ids(df)
    
    # Get feature importance
    current_dir = os.path.dirname(os.path.abspath(__file__))
    importance_path = os.path.join(current_dir, 'features', '03_lightgbm_model.csv')
    importance_df = pd.read_csv(importance_path)
    
    # Create a dictionary of weights for top features
    print("Creating importance-based feature weights...")
    # The top 10 non-user features get higher weights
    feature_weights = {}
    non_user_features = importance_df[~importance_df['Feature'].str.startswith('user_id_')]
    
    # Use the top 10 non-user features' importance values for weighting
    for _, row in non_user_features.head(10).iterrows():
        feature = row['Feature']
        # Scale feature weights: higher importance = higher weight
        # Normalize by the max importance and multiply by weight factor
        weight = 30.0 * (row['Importance'] / non_user_features['Importance'].max())
        feature_weights[feature] = weight
        print(f"  {feature}: {weight:.2f}x")
    
    # Feature weights for user IDs (reduced weight)
    user_weight = 0.3
    print(f"  user_id_*: {user_weight:.2f}x")
    
    # Other features get moderate weights
    default_weight = 10.0
    print(f"  Other features: {default_weight:.2f}x")
    
    # Make binary predictions with importance-based feature weights
    prediction_matrix, proba_matrix = make_prediction_matrix_weighted(
        df, top_user_ids, test_question_ids, model, embeddings, feature_weights, 
        threshold=0.5, 
        default_weight=default_weight,
        user_weight=user_weight
    )
    
    # Save prediction matrix
    save_prediction_matrix(prediction_matrix, test_question_ids, top_user_ids, proba_matrix)
    
    print("Binary prediction matrix generated successfully!")

if __name__ == "__main__":
    main() 
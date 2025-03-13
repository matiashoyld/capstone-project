import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split

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
    embeddings_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded from {embeddings_path}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def load_trained_model():
    """
    Load the trained neural network model and preprocessors.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', '04_neural_net_model.h5')
    preprocessors_path = os.path.join(current_dir, 'models', '04_neural_net_results.json')
    
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Try to load preprocessors from JSON if available
        try:
            with open(preprocessors_path, 'r') as f:
                results = json.load(f)
            print(f"Model results loaded from {preprocessors_path}")
        except:
            results = None
            print("No results file found. Will recreate necessary preprocessors.")
        
        return model, results
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_top_user_ids(df, n=100):
    """
    Get the top n user IDs by number of responses.
    """
    user_counts = df['user_id'].value_counts()
    top_users = user_counts.head(n).index.tolist()
    print(f"Extracted top {len(top_users)} user IDs")
    return top_users

def get_test_question_ids(df):
    """
    Get the question IDs used in the test set.
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
        test_size=0.2,
        random_state=42,
        stratify=question_df['correctness_bin']
    )
    
    test_question_ids = test_questions['question_id'].tolist()
    print(f"Retrieved {len(test_question_ids)} test question IDs")
    
    return test_question_ids

def create_user_index_mapping(df):
    """
    Create a mapping from user_id to index for embedding lookup.
    """
    all_user_ids = df['user_id'].unique()
    user_vocab_size = len(all_user_ids) + 1  # +1 for unknown/padding
    
    # Create a mapping from user_id to index
    user_id_to_index = {user_id: idx + 1 for idx, user_id in enumerate(all_user_ids)}
    
    # Add unknown user ID
    user_id_to_index[0] = 0  # 0 index for unknown/padding
    
    return user_id_to_index, user_vocab_size

def prepare_features(df, embeddings=None):
    """
    Prepare features for the neural network model.
    """
    # Feature columns to use
    text_features = [
        'title_word_count', 'title_char_count', 'title_avg_word_length',
        'title_digit_count', 'title_special_char_count', 
        'title_mathematical_symbols', 'title_latex_expressions'
    ]
    
    answer_features = [
        'jaccard_similarity_std', 'avg_option_length', 'avg_option_word_count'
    ]
    
    metadata_features = [
        'avg_steps', 'level', 'num_misconceptions', 'has_image'
    ]
    
    difficulty_features = [
        'irt_difficulty', 'original_difficulty'
    ]
    
    # Combine all numerical features
    numerical_features = text_features + answer_features + metadata_features + difficulty_features
    
    # Create user ID mapping
    user_id_to_index, user_vocab_size = create_user_index_mapping(df)
    
    # Prepare embedding features if available
    if embeddings is not None and 'formatted_embeddings' in embeddings:
        print("Processing question embeddings...")
        
        # Create a mapping from question_id to its embedding
        question_embeddings = {}
        for q_id, emb in zip(embeddings['question_ids'], embeddings['formatted_embeddings']):
            question_embeddings[q_id] = emb
            
        emb_dim = embeddings['formatted_embeddings'].shape[1]
    else:
        question_embeddings = None
        emb_dim = None
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_data = df[numerical_features].fillna(0)
    scaler.fit(numerical_data)
    
    return {
        'user_id_to_index': user_id_to_index,
        'user_vocab_size': user_vocab_size,
        'numerical_features': numerical_features,
        'scaler': scaler,
        'question_embeddings': question_embeddings,
        'embedding_dim': emb_dim
    }

def make_prediction_matrix(df, model, preprocessors, top_user_ids, test_question_ids, threshold=0.5):
    """
    Generate a prediction matrix for the specified user IDs and question IDs.
    
    Args:
        df: DataFrame with the data
        model: Trained neural network model
        preprocessors: Dictionary of preprocessors
        top_user_ids: List of user IDs to include
        test_question_ids: List of question IDs to include
        threshold: Threshold for binary predictions
        
    Returns:
        Binary prediction matrix and probability matrix
    """
    # Get necessary preprocessors
    user_id_to_index = preprocessors['user_id_to_index']
    scaler = preprocessors['scaler']
    numerical_features = preprocessors['numerical_features']
    question_embeddings = preprocessors['question_embeddings']
    
    # Initialize matrices
    n_users = len(top_user_ids)
    n_questions = len(test_question_ids)
    binary_matrix = np.zeros((n_users, n_questions))
    proba_matrix = np.zeros((n_users, n_questions))
    
    # Create a filtered dataframe with only test questions
    test_df = df[df['question_id'].isin(test_question_ids)].copy()
    
    # Get unique question info (we only need one row per question)
    question_info = test_df.drop_duplicates('question_id').set_index('question_id')
    
    # Progress bar
    progress_bar = tqdm(total=n_users * n_questions, desc="Generating predictions")
    
    # For each user and question, make a prediction
    for i, user_id in enumerate(top_user_ids):
        # Convert user ID to index for embedding lookup
        user_idx = user_id_to_index.get(user_id, 0)  # Use 0 (unknown) if not found
        
        # Prepare batch inputs for all questions for this user
        user_batch = np.array([user_idx] * n_questions)
        numerical_batch = np.zeros((n_questions, len(numerical_features)))
        embedding_batch = None
        
        if question_embeddings is not None:
            embedding_batch = np.zeros((n_questions, preprocessors['embedding_dim']))
        
        # Fill in features for each question
        for j, q_id in enumerate(test_question_ids):
            # Get numerical features for this question
            if q_id in question_info.index:
                numerical_batch[j] = question_info.loc[q_id, numerical_features].fillna(0).values
            
            # Get embedding for this question if available
            if question_embeddings is not None and q_id in question_embeddings:
                embedding_batch[j] = question_embeddings[q_id]
        
        # Scale numerical features
        numerical_batch = scaler.transform(numerical_batch)
        
        # Make predictions
        inputs = [user_batch, numerical_batch]
        if embedding_batch is not None:
            inputs.append(embedding_batch)
        
        probs = model.predict(inputs, verbose=0).flatten()
        preds = (probs > threshold).astype(int)
        
        # Store in matrices
        binary_matrix[i] = preds
        proba_matrix[i] = probs
        
        # Update progress bar
        progress_bar.update(n_questions)
    
    progress_bar.close()
    
    return binary_matrix, proba_matrix

def analyze_prediction_variation(binary_matrix, proba_matrix, top_user_ids, test_question_ids):
    """
    Analyze variation in predictions across users and questions.
    """
    # Convert to dataframes for easier analysis
    binary_df = pd.DataFrame(binary_matrix, index=top_user_ids, columns=test_question_ids)
    proba_df = pd.DataFrame(proba_matrix, index=top_user_ids, columns=test_question_ids)
    
    # Calculate statistics for binary predictions
    binary_by_question = binary_df.mean(axis=0)
    binary_by_user = binary_df.mean(axis=1)
    
    # Calculate standard deviation across questions for each user
    std_across_questions = proba_df.std(axis=1)
    
    # Calculate standard deviation across users for each question
    std_across_users = proba_df.std(axis=0)
    
    print("\nPrediction statistics:")
    print(f"Mean probability: {proba_df.values.mean():.4f}")
    print(f"Min probability: {proba_df.values.min():.4f}")
    print(f"Max probability: {proba_df.values.max():.4f}")
    
    print("\nVariation statistics:")
    print(f"Standard deviation across questions (average): {std_across_questions.mean():.6f}")
    print(f"Standard deviation across users (average): {std_across_users.mean():.6f}")
    
    # Sample the first 5 values for the first 5 users to see variation
    print("\nFirst 5 values for first 5 users:")
    print(proba_df.iloc[:5, :5].to_string(float_format='%.3f'))
    
    # Calculate the range of predictions for the first 5 questions
    print("\nRange of predictions for first 5 questions:")
    print(proba_df.iloc[:, :5].max(axis=0) - proba_df.iloc[:, :5].min(axis=0))
    
    return {
        'binary_df': binary_df,
        'proba_df': proba_df,
        'std_across_questions': std_across_questions,
        'std_across_users': std_across_users
    }

def save_prediction_matrices(binary_df, proba_df):
    """
    Save the prediction matrices to CSV files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_dir = os.path.join(current_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    binary_path = os.path.join(predictions_dir, '05_binary_prediction_matrix.csv')
    proba_path = os.path.join(predictions_dir, '05_probability_matrix.csv')
    
    binary_df.to_csv(binary_path)
    proba_df.to_csv(proba_path)
    
    print(f"Binary prediction matrix saved to {binary_path}")
    print(f"Probability matrix saved to {proba_path}")

def visualize_predictions(binary_df, proba_df, std_across_questions, std_across_users):
    """
    Create visualizations of the prediction matrices and variation statistics.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Set up the plot style
    sns.set(style="whitegrid")
    
    # Plot 1: Heatmap of probability matrix (sample)
    plt.figure(figsize=(12, 8))
    sample_size = min(50, len(proba_df))
    sns.heatmap(
        proba_df.iloc[:sample_size, :sample_size], 
        cmap='viridis', 
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Probability'}
    )
    plt.title('Probability Matrix Sample (First 50 Users x 50 Questions)')
    plt.xlabel('Question ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '05_probability_matrix_heatmap.png'))
    plt.close()
    
    # Plot 2: Heatmap of binary prediction matrix (sample)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        binary_df.iloc[:sample_size, :sample_size], 
        cmap='Blues', 
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Prediction (0/1)'}
    )
    plt.title('Binary Prediction Matrix Sample (First 50 Users x 50 Questions)')
    plt.xlabel('Question ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '05_binary_matrix_heatmap.png'))
    plt.close()
    
    # Plot 3: Distribution of standard deviations across questions
    plt.figure(figsize=(10, 6))
    sns.histplot(std_across_questions, kde=True)
    plt.title('Distribution of Standard Deviations Across Questions')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.axvline(std_across_questions.mean(), color='red', linestyle='--', 
                label=f'Mean: {std_across_questions.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '05_std_across_questions.png'))
    plt.close()
    
    # Plot 4: Distribution of standard deviations across users
    plt.figure(figsize=(10, 6))
    sns.histplot(std_across_users, kde=True)
    plt.title('Distribution of Standard Deviations Across Users')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.axvline(std_across_users.mean(), color='red', linestyle='--', 
                label=f'Mean: {std_across_users.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '05_std_across_users.png'))
    plt.close()
    
    # Plot 5: Compare variation across questions vs across users
    plt.figure(figsize=(10, 6))
    plt.boxplot([std_across_questions, std_across_users], 
                labels=['Across Questions', 'Across Users'])
    plt.title('Variation Comparison: Across Questions vs Across Users')
    plt.ylabel('Standard Deviation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '05_variation_comparison.png'))
    plt.close()
    
    print(f"Visualizations saved to {images_dir}")

def compare_with_lightgbm():
    """
    Compare neural network predictions with LightGBM predictions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nn_proba_path = os.path.join(current_dir, 'predictions', '05_probability_matrix.csv')
    lgbm_proba_path = os.path.join(current_dir, 'predictions', '04_probability_matrix.csv')
    
    try:
        nn_proba = pd.read_csv(nn_proba_path, index_col=0)
        lgbm_proba = pd.read_csv(lgbm_proba_path, index_col=0)
        
        # Check if they have matching dimensions
        if nn_proba.shape != lgbm_proba.shape:
            print("Warning: Matrices have different shapes, adjusting for comparison")
            # Find common indices and columns
            common_users = list(set(nn_proba.index).intersection(set(lgbm_proba.index)))
            common_questions = list(set(nn_proba.columns).intersection(set(lgbm_proba.columns)))
            
            # Filter to common elements
            nn_proba = nn_proba.loc[common_users, common_questions]
            lgbm_proba = lgbm_proba.loc[common_users, common_questions]
        
        # Calculate differences
        diff_matrix = nn_proba - lgbm_proba
        
        # Statistics on differences
        print("\nComparison with LightGBM predictions:")
        print(f"Mean absolute difference: {np.abs(diff_matrix).mean().mean():.4f}")
        print(f"Max absolute difference: {np.abs(diff_matrix).max().max():.4f}")
        
        # Calculate standard deviations for both models
        nn_std_questions = nn_proba.std(axis=1).mean()
        nn_std_users = nn_proba.std(axis=0).mean()
        
        lgbm_std_questions = lgbm_proba.std(axis=1).mean()
        lgbm_std_users = lgbm_proba.std(axis=0).mean()
        
        print("\nVariation comparison:")
        print(f"Neural Network - Std across questions: {nn_std_questions:.6f}")
        print(f"Neural Network - Std across users: {nn_std_users:.6f}")
        print(f"LightGBM - Std across questions: {lgbm_std_questions:.6f}")
        print(f"LightGBM - Std across users: {lgbm_std_users:.6f}")
        
        # Create visualization directory
        images_dir = os.path.join(current_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Plot difference heatmap
        plt.figure(figsize=(12, 8))
        sample_size = min(50, len(diff_matrix))
        sns.heatmap(
            diff_matrix.iloc[:sample_size, :sample_size],
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'NN - LightGBM'}
        )
        plt.title('Difference Between Neural Network and LightGBM Predictions')
        plt.xlabel('Question ID')
        plt.ylabel('User ID')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, '05_model_comparison_heatmap.png'))
        plt.close()
        
        # Plot variation comparison
        plt.figure(figsize=(10, 6))
        models = ['Neural Network', 'LightGBM']
        std_questions = [nn_std_questions, lgbm_std_questions]
        std_users = [nn_std_users, lgbm_std_users]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, std_questions, width, label='Std across questions')
        ax.bar(x + width/2, std_users, width, label='Std across users')
        
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Variation Comparison Between Models')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, '05_model_variation_comparison.png'))
        plt.close()
        
        print(f"Comparison visualizations saved to {images_dir}")
        
        return {
            'diff_matrix': diff_matrix,
            'nn_std_questions': nn_std_questions,
            'nn_std_users': nn_std_users,
            'lgbm_std_questions': lgbm_std_questions,
            'lgbm_std_users': lgbm_std_users
        }
    
    except Exception as e:
        print(f"Error comparing with LightGBM predictions: {e}")
        return None

def main():
    # Load data
    df = load_data()
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Load model
    model, results = load_trained_model()
    
    if model is None:
        print("Error: Could not load trained model")
        return
    
    # Get top user IDs and test question IDs
    top_user_ids = get_top_user_ids(df, n=100)
    test_question_ids = get_test_question_ids(df)
    
    # Prepare features and preprocessors
    preprocessors = prepare_features(df, embeddings)
    
    # Generate prediction matrices
    print("\nGenerating prediction matrices...")
    binary_matrix, proba_matrix = make_prediction_matrix(
        df, model, preprocessors, top_user_ids, test_question_ids
    )
    
    # Convert to dataframes for analysis
    binary_df = pd.DataFrame(binary_matrix, index=top_user_ids, columns=test_question_ids)
    proba_df = pd.DataFrame(proba_matrix, index=top_user_ids, columns=test_question_ids)
    
    # Analyze prediction variation
    print("\nAnalyzing prediction variation...")
    analysis_results = analyze_prediction_variation(
        binary_matrix, proba_matrix, top_user_ids, test_question_ids
    )
    
    # Save prediction matrices
    save_prediction_matrices(binary_df, proba_df)
    
    # Visualize predictions
    print("\nCreating visualizations...")
    visualize_predictions(
        analysis_results['binary_df'],
        analysis_results['proba_df'],
        analysis_results['std_across_questions'],
        analysis_results['std_across_users']
    )
    
    # Compare with LightGBM predictions
    print("\nComparing with LightGBM predictions...")
    compare_results = compare_with_lightgbm()
    
    print("\nNeural network prediction generation completed.")

if __name__ == "__main__":
    main() 
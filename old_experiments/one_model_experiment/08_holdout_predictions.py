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

def load_data():
    """
    Load the processed data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df

def load_holdout_questions():
    """
    Load the 10% holdout test set questions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_path = os.path.join(current_dir, 'data', '07_holdout_test_questions.csv')
    print(f"Loading holdout questions from {holdout_path}")
    
    holdout_df = pd.read_csv(holdout_path)
    holdout_question_ids = holdout_df['question_id'].tolist()
    
    print(f"Loaded {len(holdout_question_ids)} holdout question IDs")
    return holdout_question_ids

def load_embeddings():
    """
    Load the question embeddings.
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
    model_path = os.path.join(current_dir, 'models', '07_neural_net_holdout.h5')
    preprocessors_path = os.path.join(current_dir, 'models', '07_neural_net_holdout_preprocessors.pkl')
    
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load preprocessors
        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
        print(f"Preprocessors loaded from {preprocessors_path}")
        
        return model, preprocessors
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

def make_prediction_matrix(df, model, preprocessors, top_user_ids, holdout_question_ids, threshold=0.5):
    """
    Generate a prediction matrix for the specified user IDs and holdout question IDs.
    
    Args:
        df: DataFrame with all data
        model: Trained neural network model
        preprocessors: Dictionary of preprocessors
        top_user_ids: List of user IDs to include
        holdout_question_ids: List of holdout question IDs to include
        threshold: Threshold for binary predictions
        
    Returns:
        Binary prediction matrix and probability matrix
    """
    # Get necessary preprocessors
    user_id_to_index = preprocessors['user_id_to_index']
    scaler = preprocessors['scaler']
    numerical_features = preprocessors['numerical_features']
    
    # Initialize matrices
    n_users = len(top_user_ids)
    n_questions = len(holdout_question_ids)
    binary_matrix = np.zeros((n_users, n_questions))
    proba_matrix = np.zeros((n_users, n_questions))
    
    # Create a filtered dataframe with only holdout questions
    holdout_df = df[df['question_id'].isin(holdout_question_ids)].copy()
    
    # Get unique question info (we only need one row per question)
    question_info = holdout_df.drop_duplicates('question_id').set_index('question_id')
    
    # Check model input structure
    use_embeddings = False
    if len(model.inputs) > 2:  # If model has more than 2 inputs (user_id, numerical), it probably uses embeddings
        use_embeddings = True
        print("Model appears to use embeddings, loading...")
        embeddings = load_embeddings()
        
        # Create a mapping from question_id to its embedding
        question_embeddings = {}
        if embeddings is not None and 'formatted_embeddings' in embeddings:
            for q_id, emb in zip(embeddings['question_ids'], embeddings['formatted_embeddings']):
                question_embeddings[q_id] = emb
            embedding_dim = embeddings['formatted_embeddings'].shape[1]
            print(f"Created embeddings mapping with dimension {embedding_dim}")
        else:
            print("Warning: Could not create question embeddings mapping")
    
    # Progress bar
    progress_bar = tqdm(total=n_users * n_questions, desc="Generating holdout predictions")
    
    # For each user and question, make a prediction
    for i, user_id in enumerate(top_user_ids):
        # Convert user ID to index for embedding lookup
        user_idx = user_id_to_index.get(user_id, 0)  # Use 0 (unknown) if not found
        
        # Prepare batch inputs for all questions for this user
        user_batch = np.array([user_idx] * n_questions)
        numerical_batch = np.zeros((n_questions, len(numerical_features)))
        embedding_batch = None
        
        if use_embeddings:
            embedding_batch = np.zeros((n_questions, embedding_dim))
        
        # Fill in features for each question
        for j, q_id in enumerate(holdout_question_ids):
            # Get numerical features for this question
            if q_id in question_info.index:
                # Handle potential missing values
                features = question_info.loc[q_id, numerical_features].copy()
                for col in numerical_features:
                    if pd.isna(features[col]):
                        features[col] = 0  # Replace NaN with 0
                numerical_batch[j] = features.values
            
            # Get embedding for this question if available
            if use_embeddings and q_id in question_embeddings:
                embedding_batch[j] = question_embeddings[q_id]
        
        # Scale numerical features
        numerical_batch = scaler.transform(numerical_batch)
        
        # Make predictions
        inputs = [user_batch, numerical_batch]
        if use_embeddings:
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

def analyze_prediction_variation(binary_matrix, proba_matrix, top_user_ids, holdout_question_ids):
    """
    Analyze variation in predictions across users and questions.
    """
    # Convert to dataframes for easier analysis
    binary_df = pd.DataFrame(binary_matrix, index=top_user_ids, columns=holdout_question_ids)
    proba_df = pd.DataFrame(proba_matrix, index=top_user_ids, columns=holdout_question_ids)
    
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
    
    binary_path = os.path.join(predictions_dir, '08_holdout_binary_prediction_matrix.csv')
    proba_path = os.path.join(predictions_dir, '08_holdout_probability_matrix.csv')
    
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
    sample_size_rows = min(50, len(proba_df))
    sample_size_cols = min(50, proba_df.shape[1])
    sns.heatmap(
        proba_df.iloc[:sample_size_rows, :sample_size_cols], 
        cmap='viridis', 
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Probability'}
    )
    plt.title('Holdout Probability Matrix Sample (First 50 Users x 50 Questions)')
    plt.xlabel('Question ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '08_holdout_probability_matrix_heatmap.png'))
    plt.close()
    
    # Plot 2: Heatmap of binary prediction matrix (sample)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        binary_df.iloc[:sample_size_rows, :sample_size_cols], 
        cmap='Blues', 
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Prediction (0/1)'}
    )
    plt.title('Holdout Binary Prediction Matrix Sample (First 50 Users x 50 Questions)')
    plt.xlabel('Question ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '08_holdout_binary_matrix_heatmap.png'))
    plt.close()
    
    # Plot 3: Distribution of standard deviations across questions
    plt.figure(figsize=(10, 6))
    sns.histplot(std_across_questions, kde=True)
    plt.title('Distribution of Standard Deviations Across Questions (Holdout Set)')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.axvline(std_across_questions.mean(), color='red', linestyle='--', 
                label=f'Mean: {std_across_questions.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '08_holdout_std_across_questions.png'))
    plt.close()
    
    # Plot 4: Distribution of standard deviations across users
    plt.figure(figsize=(10, 6))
    sns.histplot(std_across_users, kde=True)
    plt.title('Distribution of Standard Deviations Across Users (Holdout Set)')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.axvline(std_across_users.mean(), color='red', linestyle='--', 
                label=f'Mean: {std_across_users.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '08_holdout_std_across_users.png'))
    plt.close()
    
    # Plot 5: Compare variation across questions vs across users
    plt.figure(figsize=(10, 6))
    plt.boxplot([std_across_questions, std_across_users], 
                labels=['Across Questions', 'Across Users'])
    plt.title('Variation Comparison: Across Questions vs Across Users (Holdout Set)')
    plt.ylabel('Standard Deviation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, '08_holdout_variation_comparison.png'))
    plt.close()
    
    print(f"Visualizations saved to {images_dir}")

def compare_with_training_set():
    """
    Compare holdout predictions with training set predictions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_proba_path = os.path.join(current_dir, 'predictions', '08_holdout_probability_matrix.csv')
    training_proba_path = os.path.join(current_dir, 'predictions', '05_probability_matrix.csv')
    
    try:
        holdout_proba = pd.read_csv(holdout_proba_path, index_col=0)
        training_proba = pd.read_csv(training_proba_path, index_col=0)
        
        # Calculate statistics for both sets
        holdout_std_questions = holdout_proba.std(axis=1).mean()
        holdout_std_users = holdout_proba.std(axis=0).mean()
        
        training_std_questions = training_proba.std(axis=1).mean()
        training_std_users = training_proba.std(axis=0).mean()
        
        print("\nComparison with training set predictions:")
        print(f"Holdout - Std across questions: {holdout_std_questions:.6f}")
        print(f"Holdout - Std across users: {holdout_std_users:.6f}")
        print(f"Training - Std across questions: {training_std_questions:.6f}")
        print(f"Training - Std across users: {training_std_users:.6f}")
        
        # Create visualization directory
        images_dir = os.path.join(current_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Plot variation comparison
        plt.figure(figsize=(10, 8))
        
        # Bar chart of standard deviations
        labels = ['Std across questions', 'Std across users']
        holdout_values = [holdout_std_questions, holdout_std_users]
        training_values = [training_std_questions, training_std_users]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, holdout_values, width, label='Holdout Set')
        ax.bar(x + width/2, training_values, width, label='Training Set')
        
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Prediction Variation: Holdout vs Training Set')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, '08_holdout_vs_training_variation.png'))
        plt.close()
        
        print(f"Comparison visualization saved to {os.path.join(images_dir, '08_holdout_vs_training_variation.png')}")
        
        return {
            'holdout_std_questions': holdout_std_questions,
            'holdout_std_users': holdout_std_users,
            'training_std_questions': training_std_questions,
            'training_std_users': training_std_users
        }
    
    except Exception as e:
        print(f"Error comparing with training set predictions: {e}")
        return None

def main():
    # Load data
    df = load_data()
    
    # Load holdout question IDs
    holdout_question_ids = load_holdout_questions()
    
    # Load trained model and preprocessors
    model, preprocessors = load_trained_model()
    if model is None or preprocessors is None:
        print("Error: Could not load model or preprocessors. Exiting.")
        return
    
    # Get top user IDs
    top_user_ids = get_top_user_ids(df, n=100)
    
    # Generate prediction matrices
    print("\nGenerating prediction matrices for holdout questions...")
    binary_matrix, proba_matrix = make_prediction_matrix(
        df, model, preprocessors, top_user_ids, holdout_question_ids
    )
    
    # Convert to dataframes for analysis
    binary_df = pd.DataFrame(binary_matrix, index=top_user_ids, columns=holdout_question_ids)
    proba_df = pd.DataFrame(proba_matrix, index=top_user_ids, columns=holdout_question_ids)
    
    # Analyze prediction variation
    print("\nAnalyzing prediction variation for holdout questions...")
    analysis_results = analyze_prediction_variation(
        binary_matrix, proba_matrix, top_user_ids, holdout_question_ids
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
    
    # Compare with training set predictions
    print("\nComparing with training set predictions...")
    comparison_results = compare_with_training_set()
    
    print("\nHoldout prediction generation completed.")

if __name__ == "__main__":
    main() 
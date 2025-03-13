import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import json
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU memory growth to avoid memory allocation errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is available: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU available, using CPU")

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

def stratified_question_split_3way(df, test_size=0.1, val_size=0.2, random_state=42):
    """
    Split questions into train, validation, and test sets in a stratified manner 
    based on correct answer ratios and difficulty to ensure balanced distributions.
    
    Args:
        df: DataFrame with all data
        test_size: Proportion of questions for the test set (default: 0.1 or 10%)
        val_size: Proportion of questions for the validation set (default: 0.2 or 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Lists of question_ids for train, validation, and test sets
    """
    # Get unique questions and their metrics
    question_df = df.groupby('question_id').agg({
        'is_correct': 'mean',  # Average correctness rate per question
        'irt_difficulty': 'first'  # IRT difficulty for stratification
    }).reset_index()
    
    # Create bins for both correctness and difficulty for multi-level stratification
    question_df['correctness_bin'] = pd.qcut(question_df['is_correct'], 5, labels=False)
    
    # Handle cases where irt_difficulty might be missing
    if question_df['irt_difficulty'].isna().any():
        question_df['irt_difficulty'].fillna(question_df['irt_difficulty'].median(), inplace=True)
    
    question_df['difficulty_bin'] = pd.qcut(question_df['irt_difficulty'], 5, labels=False)
    
    # Combine bins to create stratification target
    question_df['strat_target'] = question_df['correctness_bin'].astype(str) + '_' + question_df['difficulty_bin'].astype(str)
    
    # First split: separate out the test set (10%)
    train_val_questions, test_questions = train_test_split(
        question_df,
        test_size=test_size,
        random_state=random_state,
        stratify=question_df['strat_target']
    )
    
    # Second split: split the remaining data into train (70%) and validation (20%)
    # val_size_adjusted = val_size / (1 - test_size)  # recalculate to get 20% of the original data
    train_questions, val_questions = train_test_split(
        train_val_questions,
        test_size=val_size / (1 - test_size),  # Adjust to get the right proportion
        random_state=random_state,
        stratify=train_val_questions['strat_target']
    )
    
    print(f"Train set: {len(train_questions)} questions ({len(train_questions)/len(question_df):.1%})")
    print(f"Validation set: {len(val_questions)} questions ({len(val_questions)/len(question_df):.1%})")
    print(f"Test set: {len(test_questions)} questions ({len(test_questions)/len(question_df):.1%})")
    
    # Verify distributions
    print("\nCorrectness rate distribution:")
    print("Train mean: {:.4f}, std: {:.4f}".format(
        train_questions['is_correct'].mean(), 
        train_questions['is_correct'].std()
    ))
    print("Validation mean: {:.4f}, std: {:.4f}".format(
        val_questions['is_correct'].mean(), 
        val_questions['is_correct'].std()
    ))
    print("Test mean: {:.4f}, std: {:.4f}".format(
        test_questions['is_correct'].mean(), 
        test_questions['is_correct'].std()
    ))
    
    print("\nDifficulty distribution:")
    print("Train mean: {:.4f}, std: {:.4f}".format(
        train_questions['irt_difficulty'].mean(), 
        train_questions['irt_difficulty'].std()
    ))
    print("Validation mean: {:.4f}, std: {:.4f}".format(
        val_questions['irt_difficulty'].mean(), 
        val_questions['irt_difficulty'].std()
    ))
    print("Test mean: {:.4f}, std: {:.4f}".format(
        test_questions['irt_difficulty'].mean(), 
        test_questions['irt_difficulty'].std()
    ))
    
    # Save the test set questions to CSV for future reference
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    test_questions_path = os.path.join(data_dir, '07_holdout_test_questions.csv')
    test_questions.to_csv(test_questions_path, index=False)
    print(f"\nSaved {len(test_questions)} test questions to {test_questions_path}")
    
    return train_questions['question_id'].tolist(), val_questions['question_id'].tolist(), test_questions['question_id'].tolist()

def prepare_datasets(df, train_question_ids, val_question_ids, embeddings=None, user_embedding_dim=8):
    """
    Prepare datasets for the neural network model.
    
    Args:
        df: DataFrame with all data
        train_question_ids: List of question IDs for training
        val_question_ids: List of question IDs for validation
        embeddings: Dictionary of question embeddings
        user_embedding_dim: Dimension of user embeddings
        
    Returns:
        Training and validation datasets, preprocessors
    """
    # Split data based on question IDs
    train_df = df[df['question_id'].isin(train_question_ids)].copy()
    val_df = df[df['question_id'].isin(val_question_ids)].copy()
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    
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
    
    # Process user IDs - create a vocabulary for embedding
    # Get all unique user IDs from both train and validation sets
    all_user_ids = np.unique(np.concatenate([train_df['user_id'].unique(), val_df['user_id'].unique()]))
    user_vocab_size = len(all_user_ids) + 1  # +1 for unknown/padding
    
    # Create a mapping from user_id to index
    user_id_to_index = {user_id: idx + 1 for idx, user_id in enumerate(all_user_ids)}
    
    # Add unknown user ID
    user_id_to_index[0] = 0  # 0 index for unknown/padding
    
    # Convert user IDs to indices
    train_df['user_id_idx'] = train_df['user_id'].map(user_id_to_index)
    val_df['user_id_idx'] = val_df['user_id'].map(user_id_to_index)
    
    # Check if any user IDs were not found
    if train_df['user_id_idx'].isna().any() or val_df['user_id_idx'].isna().any():
        print("Warning: Some user IDs were not found in the mapping.")
        # Fill NaN values with 0 (unknown user)
        train_df['user_id_idx'] = train_df['user_id_idx'].fillna(0).astype(int)
        val_df['user_id_idx'] = val_df['user_id_idx'].fillna(0).astype(int)
    else:
        train_df['user_id_idx'] = train_df['user_id_idx'].astype(int)
        val_df['user_id_idx'] = val_df['user_id_idx'].astype(int)
    
    # Process skills - extract and convert to a list of skill IDs
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
    
    # Process skills
    train_df['skills_list'] = train_df['skills'].apply(process_skills)
    val_df['skills_list'] = val_df['skills'].apply(process_skills)
    
    # Extract all unique skill IDs
    all_skills = set()
    for skills in train_df['skills_list'].values:
        all_skills.update(skills)
    for skills in val_df['skills_list'].values:
        all_skills.update(skills)
    
    skill_vocab_size = max(all_skills) + 1 if all_skills else 1
    
    # Prepare embedding features if available
    if embeddings is not None and 'formatted_embeddings' in embeddings:
        print("Processing question embeddings...")
        
        # Create a mapping from question_id to its embedding
        question_embeddings = {}
        for q_id, emb in zip(embeddings['question_ids'], embeddings['formatted_embeddings']):
            question_embeddings[q_id] = emb
        
        # Add embeddings to dataframes
        def get_embedding(q_id):
            return question_embeddings.get(q_id, np.zeros(embeddings['formatted_embeddings'].shape[1]))
        
        # Apply embedding lookup
        train_embeddings = np.array([get_embedding(q_id) for q_id in train_df['question_id']])
        val_embeddings = np.array([get_embedding(q_id) for q_id in val_df['question_id']])
        
        print(f"Embedding shape: {train_embeddings.shape[1]} dimensions")
    else:
        train_embeddings = None
        val_embeddings = None
        print("No embeddings available")
    
    # Handle missing values in numerical features
    for feature in numerical_features:
        if train_df[feature].isna().any() or val_df[feature].isna().any():
            # Calculate median from train set
            train_median = train_df[feature].median()
            # Apply to both train and validation
            train_df[feature] = train_df[feature].fillna(train_median)
            val_df[feature] = val_df[feature].fillna(train_median)
    
    # Normalize numerical features
    scaler = StandardScaler()
    train_numerical = train_df[numerical_features]
    val_numerical = val_df[numerical_features]
    
    # Fit scaler on training data only
    train_numerical_scaled = scaler.fit_transform(train_numerical)
    val_numerical_scaled = scaler.transform(val_numerical)
    
    # Prepare labels
    train_labels = train_df['is_correct'].astype(int).values
    val_labels = val_df['is_correct'].astype(int).values
    
    # Create dataset dictionaries
    train_dataset = {
        'user_id': train_df['user_id_idx'].values,
        'numerical_features': train_numerical_scaled,
        'question_id': train_df['question_id'].values,
        'skills': train_df['skills_list'].values,
    }
    
    val_dataset = {
        'user_id': val_df['user_id_idx'].values,
        'numerical_features': val_numerical_scaled,
        'question_id': val_df['question_id'].values,
        'skills': val_df['skills_list'].values,
    }
    
    # Add embeddings if available
    if train_embeddings is not None:
        train_dataset['embeddings'] = train_embeddings
        val_dataset['embeddings'] = val_embeddings
    
    # Get question IDs and user IDs for reference
    train_question_ids_actual = train_df['question_id'].values
    val_question_ids_actual = val_df['question_id'].values
    train_user_ids = train_df['user_id'].values
    val_user_ids = val_df['user_id'].values
    
    # Store preprocessors
    preprocessors = {
        'scaler': scaler,
        'user_id_to_index': user_id_to_index,
        'numerical_features': numerical_features,
        'user_vocab_size': user_vocab_size,
        'skill_vocab_size': skill_vocab_size
    }
    
    return (train_dataset, train_labels, train_question_ids_actual, train_user_ids,
            val_dataset, val_labels, val_question_ids_actual, val_user_ids,
            preprocessors)

def create_model(user_vocab_size, numerical_feature_size, embedding_size=None,
                skill_vocab_size=None, user_embedding_dim=8, l2_reg=0.001):
    """
    Create a neural network model with user embeddings and numerical features.
    
    Args:
        user_vocab_size: Size of user vocabulary (number of unique users + 1)
        numerical_feature_size: Number of numerical features
        embedding_size: Size of question embedding features (if using)
        skill_vocab_size: Size of skill vocabulary (if using skill features)
        user_embedding_dim: Dimension of user embeddings
        l2_reg: L2 regularization strength
        
    Returns:
        Compiled model
    """
    # Input layers
    user_input = tf.keras.Input(shape=(1,), name='user_input')
    numerical_input = tf.keras.Input(shape=(numerical_feature_size,), name='numerical_input')
    
    # User embedding layer
    user_embedding = layers.Embedding(
        user_vocab_size,
        user_embedding_dim,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='user_embedding'
    )(user_input)
    user_embedding = layers.Flatten(name='flatten_user_embedding')(user_embedding)
    
    # Process numerical features
    numerical_features = layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='numerical_dense1'
    )(numerical_input)
    
    # Add question embeddings if available
    inputs = [user_input, numerical_input]
    features_to_concat = [user_embedding, numerical_features]
    
    if embedding_size is not None:
        embedding_input = tf.keras.Input(shape=(embedding_size,), name='embedding_input')
        inputs.append(embedding_input)
        
        # Process embeddings
        embedding_features = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='embedding_dense'
        )(embedding_input)
        
        features_to_concat.append(embedding_features)
    
    # Combine all features
    concat_features = layers.Concatenate(name='concat_features')(features_to_concat)
    
    # Hidden layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense1')(concat_features)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_and_evaluate_model(train_dataset, train_labels, val_dataset, val_labels, preprocessors,
                           embedding_size=None, batch_size=256, epochs=50):
    """
    Train and evaluate the neural network model.
    
    Args:
        train_dataset: Dictionary of training data features
        train_labels: Training labels
        val_dataset: Dictionary of validation data features
        val_labels: Validation labels
        preprocessors: Dictionary of preprocessors
        embedding_size: Size of question embedding features (if using)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        
    Returns:
        Trained model and evaluation metrics
    """
    # Create model
    model = create_model(
        user_vocab_size=preprocessors['user_vocab_size'],
        numerical_feature_size=train_dataset['numerical_features'].shape[1],
        embedding_size=embedding_size,
        skill_vocab_size=preprocessors['skill_vocab_size'],
        user_embedding_dim=8,
        l2_reg=0.001
    )
    
    # Model summary
    model.summary()
    
    # Prepare inputs for training and validation
    train_inputs = [train_dataset['user_id'], train_dataset['numerical_features']]
    val_inputs = [val_dataset['user_id'], val_dataset['numerical_features']]
    
    if embedding_size is not None and 'embeddings' in train_dataset:
        train_inputs.append(train_dataset['embeddings'])
        val_inputs.append(val_dataset['embeddings'])
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )
    
    # Create directory for saving model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(model_dir, '07_neural_net_holdout.h5'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=2
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    val_loss, val_accuracy, val_auc = model.evaluate(val_inputs, val_labels, verbose=0)
    
    # Get predictions
    val_probs = model.predict(val_inputs, verbose=0).flatten()
    val_preds = (val_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    auc = roc_auc_score(val_labels, val_probs)
    
    print("\nModel performance on validation set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory for saving figures
    fig_dir = os.path.join(current_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.savefig(os.path.join(fig_dir, '07_neural_net_holdout_training.png'))
    print(f"\nTraining plot saved to {os.path.join(fig_dir, '07_neural_net_holdout_training.png')}")
    
    # Create a dictionary to store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'training_time': training_time,
        'training_history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'auc': history.history['auc'],
            'val_auc': history.history['val_auc']
        }
    }
    
    # Save results to JSON
    with open(os.path.join(current_dir, 'models', '07_neural_net_holdout_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save preprocessors to allow future use with the holdout set
    with open(os.path.join(current_dir, 'models', '07_neural_net_holdout_preprocessors.pkl'), 'wb') as f:
        pickle.dump(preprocessors, f)
    
    print(f"Model and preprocessors saved to {os.path.join(current_dir, 'models')}")
    
    return model, results

def analyze_user_embeddings(model, preprocessors, save_plot=True):
    """
    Extract and analyze user embeddings from the trained model.
    
    Args:
        model: Trained model
        preprocessors: Dictionary of preprocessors
        save_plot: Whether to save the user embedding plot
    """
    # Get user embeddings
    user_embedding_layer = model.get_layer('user_embedding')
    user_embeddings = user_embedding_layer.get_weights()[0]
    
    # Map indices back to user IDs
    index_to_user_id = {idx: user_id for user_id, idx in preprocessors['user_id_to_index'].items()}
    
    # Create a dataframe for the embeddings
    embedding_df = pd.DataFrame(user_embeddings, columns=[f'dim_{i}' for i in range(user_embeddings.shape[1])])
    embedding_df['user_id_idx'] = embedding_df.index
    embedding_df['user_id'] = embedding_df['user_id_idx'].map(index_to_user_id)
    
    # Skip the padding/unknown user (idx 0)
    embedding_df = embedding_df[embedding_df['user_id_idx'] > 0]
    
    print(f"\nUser embedding shape: {user_embeddings.shape}")
    print(f"Extracted embeddings for {len(embedding_df)} users")
    
    # Compute 2D PCA or t-SNE for visualization if there are more than 2 dimensions
    if user_embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Apply PCA
        pca = PCA(n_components=2)
        embedding_2d_pca = pca.fit_transform(embedding_df[[f'dim_{i}' for i in range(user_embeddings.shape[1])]])
        embedding_df['pca_x'] = embedding_2d_pca[:, 0]
        embedding_df['pca_y'] = embedding_2d_pca[:, 1]
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Apply t-SNE if there are enough samples
        if len(embedding_df) > 50:
            tsne = TSNE(n_components=2, random_state=42)
            embedding_2d_tsne = tsne.fit_transform(embedding_df[[f'dim_{i}' for i in range(user_embeddings.shape[1])]])
            embedding_df['tsne_x'] = embedding_2d_tsne[:, 0]
            embedding_df['tsne_y'] = embedding_2d_tsne[:, 1]
        
        # Plot the embeddings
        if save_plot:
            plt.figure(figsize=(16, 7))
            
            plt.subplot(1, 2, 1)
            plt.scatter(embedding_df['pca_x'], embedding_df['pca_y'], alpha=0.7)
            plt.title('User Embeddings (PCA)')
            plt.xlabel('PCA Dimension 1')
            plt.ylabel('PCA Dimension 2')
            
            if 'tsne_x' in embedding_df.columns:
                plt.subplot(1, 2, 2)
                plt.scatter(embedding_df['tsne_x'], embedding_df['tsne_y'], alpha=0.7)
                plt.title('User Embeddings (t-SNE)')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
            
            plt.tight_layout()
            
            # Save the plot
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fig_dir = os.path.join(current_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            
            plt.savefig(os.path.join(fig_dir, '07_user_embeddings_holdout.png'))
            print(f"User embedding plot saved to {os.path.join(fig_dir, '07_user_embeddings_holdout.png')}")
    
    # Save the embeddings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    embedding_df.to_csv(os.path.join(output_dir, '07_user_embeddings_holdout.csv'), index=False)
    print(f"User embeddings saved to {os.path.join(output_dir, '07_user_embeddings_holdout.csv')}")
    
    return embedding_df

def main():
    # Load data
    df = load_data()
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Split questions into train, validation, and test sets
    train_question_ids, val_question_ids, test_question_ids = stratified_question_split_3way(df)
    
    # Prepare datasets
    (train_dataset, train_labels, train_question_ids_actual, train_user_ids,
     val_dataset, val_labels, val_question_ids_actual, val_user_ids,
     preprocessors) = prepare_datasets(df, train_question_ids, val_question_ids, embeddings)
    
    # Determine embedding size if embeddings are available
    embedding_size = None
    if 'embeddings' in train_dataset:
        embedding_size = train_dataset['embeddings'].shape[1]
    
    # Train and evaluate model
    model, results = train_and_evaluate_model(
        train_dataset, train_labels, val_dataset, val_labels, preprocessors,
        embedding_size=embedding_size, batch_size=1024, epochs=50
    )
    
    # Analyze user embeddings
    user_embeddings = analyze_user_embeddings(model, preprocessors)
    
    print("\nNeural network model training with holdout test set completed.")
    print(f"Holdout test set with {len(test_question_ids)} questions saved for future evaluation.")

if __name__ == "__main__":
    main() 
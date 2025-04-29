#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Model for Question Difficulty Prediction (Without Response Counts)
This script implements a neural network approach to predict question difficulty,
filtering out extremely easy questions (difficulty < -6) and avoiding the use of
response count features that would not be available for new questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import ast  # For safely evaluating the skills string representation
import warnings

# Neural network libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QuestionDataset(Dataset):
    """Dataset for question features and embeddings"""
    
    def __init__(self, features, embeddings, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.embeddings[idx], self.targets[idx]
        else:
            return self.features[idx], self.embeddings[idx]

class NeuralDifficultyModel(nn.Module):
    """Neural network model for predicting question difficulty"""
    
    def __init__(self, features_dim, embedding_dim, hidden_dim=128, dropout_rate=0.3):
        super(NeuralDifficultyModel, self).__init__()
        
        # Branch for processing standard features
        self.features_branch = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate)
        )
        
        # Branch for processing embeddings
        self.embeddings_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate)
        )
        
        # Combining branches for final prediction
        combined_dim = hidden_dim  # This is the sum of both branch outputs (hidden_dim // 2 each)
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features, embeddings):
        features_out = self.features_branch(features)
        embeddings_out = self.embeddings_branch(embeddings)
        combined = torch.cat((features_out, embeddings_out), dim=1)
        return self.combined_layers(combined).squeeze()

def load_data(difficulty_threshold=-6):
    """Load and merge the question data and embeddings, filtering out very easy questions"""
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
    
    # Filter out extremely easy questions
    total_before = len(questions_df)
    questions_df = questions_df[questions_df['irt_difficulty'] >= difficulty_threshold]
    filtered_count = total_before - len(questions_df)
    print(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
    
    # Load embeddings
    with open('questions_embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    
    # Merge dataframes on question_id
    merged_df = questions_df.merge(embeddings_df, on='question_id', how='inner')
    
    print(f"Loaded {len(merged_df)} questions with embeddings (after filtering)")
    print("\nSample of the loaded data:")
    print(merged_df.head())
    print("\nData info:")
    print(merged_df.info())
    
    return merged_df

def preprocess_features(df):
    """Preprocess features for model training, excluding count features"""
    print("Preprocessing features...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Extract embeddings
    embeddings = np.array(df_processed['embedding'].tolist())
    
    # Process skills
    mlb = MultiLabelBinarizer()
    skills_lists = df_processed['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    skills_binary = mlb.fit_transform(skills_lists)
    skills_df = pd.DataFrame(skills_binary, columns=mlb.classes_)
    
    # Convert all column names to strings
    skills_df.columns = skills_df.columns.astype(str)
    
    # Basic features without count features
    basic_features = ['level', 'num_misconceptions', 'has_image', 'avg_steps']
    
    # Removed count features: 'count_a', 'count_b', 'count_c', 'count_d', 'count_e', 'total_count'
    print("Note: Removed response count features which wouldn't be available for new questions")
    
    # One-hot encode categorical ID features
    id_cols = ['topic_id', 'subject_id', 'axis_id']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_ids = encoder.fit_transform(df_processed[id_cols])
    id_feature_names = [f"{col}_{val}" for col, vals in zip(id_cols, encoder.categories_) 
                        for val in vals]
    id_features_df = pd.DataFrame(encoded_ids, columns=id_feature_names)
    
    # Combine features
    X_without_embeddings = pd.concat([
        df_processed[basic_features].reset_index(drop=True),
        skills_df.reset_index(drop=True),
        id_features_df.reset_index(drop=True)
    ], axis=1)
    
    # Convert all column names to strings to avoid validation errors
    X_without_embeddings.columns = X_without_embeddings.columns.astype(str)
    
    # Scale features
    scaler = StandardScaler()
    X_without_embeddings_scaled = scaler.fit_transform(X_without_embeddings)
    
    # Target variable
    y = df_processed['irt_difficulty'].values
    
    # Save feature processors for later use
    processors = {
        'skills_mlb': mlb,
        'scaler': scaler,
        'id_encoder': encoder
    }
    
    with open('feature_processors_neural_filtered_no_counts.pkl', 'wb') as f:
        pickle.dump(processors, f)
    
    return X_without_embeddings_scaled, embeddings, y

def train_model(X, y, embeddings, batch_size=64, epochs=50, learning_rate=0.001, 
               patience=10, weight_decay=1e-5, hidden_dim=128, dropout_rate=0.3, 
               cv_folds=5):
    """Train and evaluate the neural network model"""
    print("Training neural network model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, emb_train, emb_test = train_test_split(
        X, y, embeddings, test_size=0.2, random_state=42
    )
    
    # Cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nTraining fold {fold}/{cv_folds}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        emb_fold_train, emb_fold_val = emb_train[train_idx], emb_train[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = QuestionDataset(X_fold_train, emb_fold_train, y_fold_train)
        val_dataset = QuestionDataset(X_fold_val, emb_fold_val, y_fold_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        features_dim = X.shape[1]
        embedding_dim = embeddings.shape[1]
        
        model = NeuralDifficultyModel(
            features_dim=features_dim, 
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Early stopping variables
        best_val_rmse = float('inf')
        best_epoch = 0
        no_improve_count = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            
            for features, embs, targets in train_loader:
                features, embs, targets = features.to(device), embs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features, embs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * features.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for features, embs, targets in val_loader:
                    features, embs = features.to(device), embs.to(device)
                    outputs = model(features, embs)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.numpy())
            
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            
            # Check for improvement
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print(f"Fold {fold} final validation RMSE: {best_val_rmse:.4f}")
        cv_scores.append(best_val_rmse)
    
    print(f"Cross-validation RMSE scores: {cv_scores}")
    mean_cv_rmse = np.mean(cv_scores)
    std_cv_rmse = np.std(cv_scores)
    print(f"Mean CV RMSE: {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    
    # Train final model on all training data
    print("\nTraining final model on all training data...")
    train_dataset = QuestionDataset(X_train, emb_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    features_dim = X.shape[1]
    embedding_dim = embeddings.shape[1]
    
    final_model = NeuralDifficultyModel(
        features_dim=features_dim, 
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        final_model.train()
        train_loss = 0.0
        
        for features, embs, targets in train_loader:
            features, embs, targets = features.to(device), embs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(features, embs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")
    
    # Evaluate on test data
    final_model.eval()
    test_dataset = QuestionDataset(X_test, emb_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features, embs, targets in test_loader:
            features, embs = features.to(device), embs.to(device)
            outputs = final_model(features, embs)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(targets.numpy())
    
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save the model
    torch.save(final_model.state_dict(), 'neural_question_model_filtered_no_counts.pt')
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Actual vs Predicted Difficulty (Neural Model, Filtered, No Count Features)')
    plt.savefig('neural_prediction_scatter_filtered_no_counts.png')
    
    # Save metrics to a text file
    with open('neural_model_filtered_no_counts_metrics.txt', 'w') as f:
        f.write(f"Mean CV RMSE: {mean_cv_rmse:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f}\n")
        f.write(f"Test MAE: {test_mae:.4f}\n")
        f.write(f"Test R²: {test_r2:.4f}\n")
    
    # Update metrics comparison file
    try:
        with open('metrics_comparison.txt', 'r') as f:
            content = f.read()
        
        # Create new entry for neural model without count features
        new_entry = (
            f"\n### Neural Network Model with all-MiniLM-L6-v2 embeddings (Filtered: difficulty >= -6, No Count Features)\n"
            f"- Mean CV RMSE: {mean_cv_rmse:.4f}\n"
            f"- Test RMSE: {test_rmse:.4f}\n"
            f"- Test MAE: {test_mae:.4f}\n"
            f"- Test R²: {test_r2:.4f}\n"
            f"- Note: This model excludes questions with difficulty < -6 and does not use response count features\n"
        )
        
        # Find the position to insert the new entry (under Neural Network Models section)
        nn_section = content.find("## Neural Network Models")
        next_section = content.find("##", nn_section + 1)
        
        if nn_section >= 0 and next_section >= 0:
            updated_content = content[:next_section] + new_entry + content[next_section:]
            
            with open('metrics_comparison.txt', 'w') as f:
                f.write(updated_content)
            
            print("Updated metrics_comparison.txt with neural model (no count features) results.")
        else:
            print("Could not update metrics_comparison.txt. Section markers not found.")
        
    except Exception as e:
        print(f"Error updating metrics comparison: {str(e)}")
    
    return final_model, mean_cv_rmse, test_rmse, test_mae, test_r2

def main():
    """Main function to run the neural model training pipeline without count features"""
    
    # Load the data with filtering of very easy questions
    df = load_data(difficulty_threshold=-6)
    
    # Print statistics for the target variable
    target_stats = df['irt_difficulty'].describe()
    print(f"\nTarget variable statistics after filtering:")
    print(f"Min: {target_stats['min']:.4f}, Max: {target_stats['max']:.4f}, "
          f"Mean: {target_stats['mean']:.4f}, Std: {target_stats['std']:.4f}")
    
    # Preprocess features and split data
    X, embeddings, y = preprocess_features(df)
    
    # Train and evaluate the model
    model, mean_cv_rmse, test_rmse, test_mae, test_r2 = train_model(
        X, y, embeddings, 
        batch_size=64, 
        epochs=50, 
        learning_rate=0.001,
        patience=10,
        weight_decay=1e-5,
        hidden_dim=128,
        dropout_rate=0.3,
        cv_folds=5
    )
    
    print("\nNeural model training completed. Files saved:")
    print("- neural_question_model_filtered_no_counts.pt: The trained model")
    print("- feature_processors_neural_filtered_no_counts.pkl: Feature processing objects")
    print("- neural_prediction_scatter_filtered_no_counts.png: Scatter plot of actual vs predicted values")
    print("- neural_model_filtered_no_counts_metrics.txt: Metrics summary")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 
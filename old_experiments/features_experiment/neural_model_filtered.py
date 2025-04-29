#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Model for Question Difficulty Prediction
This script implements a neural network approach to predict question difficulty,
filtering out extremely easy questions (difficulty < -6).
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return self.features[idx], self.embeddings[idx]

class NeuralDifficultyModel(nn.Module):
    """Neural network model for difficulty prediction"""
    def __init__(self, features_dim, embedding_dim, hidden_dim=128, dropout_rate=0.3):
        super(NeuralDifficultyModel, self).__init__()
        
        # Embedding processing branch
        self.emb_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Structured features processing branch
        self.feature_branch = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Combined layers
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, features, embeddings):
        emb_out = self.emb_branch(embeddings)
        feat_out = self.feature_branch(features)
        combined = torch.cat((emb_out, feat_out), dim=1)
        return self.combined(combined).squeeze()

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
    return merged_df

def preprocess_features(df):
    """Process all features for model training - matching model_training.py"""
    print("Preprocessing features...")
    
    # Convert skills from string representation to actual lists if needed
    if 'skills' in df.columns and isinstance(df['skills'].iloc[0], str):
        df['skills'] = df['skills'].apply(ast.literal_eval)
    
    # Process numeric features - same as model_training.py
    numeric_features = ['avg_steps', 'num_misconceptions', 'level']
    X_numeric = df[numeric_features].copy()
    
    # Process ID features (one-hot encoding)
    id_features = ['topic_id', 'subject_id', 'axis_id']
    X_ids = df[id_features].copy()
    
    # Handle missing values
    X_numeric.fillna(0, inplace=True)
    X_ids.fillna(-1, inplace=True)  # Use -1 for missing IDs
    
    # One-hot encode the ID features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ids_encoded = encoder.fit_transform(X_ids)
    id_feature_names = []
    for i, feature in enumerate(id_features):
        for j in range(len(encoder.categories_[i])):
            id_feature_names.append(f"{feature}_{encoder.categories_[i][j]}")
    X_ids_encoded = pd.DataFrame(X_ids_encoded, columns=id_feature_names)
    
    # Normalize numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_features)
    
    # Process binary features
    X_binary = df[['has_image']].astype(int)
    
    # Process skills (one-hot encoding) - same as model_training.py
    mlb = MultiLabelBinarizer()
    X_skills = mlb.fit_transform(df['skills'])
    skill_cols = [f'skill_{i}' for i in mlb.classes_]
    X_skills_df = pd.DataFrame(X_skills, columns=skill_cols)
    
    # Process embeddings
    X_embeddings = np.array(df['embedding'].tolist())
    
    # Save the feature processors for later use
    with open('feature_processors_neural_filtered.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'mlb': mlb,
            'encoder': encoder,
        }, f)
    
    # Combine all processed features
    X_combined = pd.concat([
        X_numeric_scaled, 
        X_binary,
        X_skills_df,
        X_ids_encoded  # Add one-hot encoded ID features
    ], axis=1)
    
    return X_combined, X_embeddings

def train_model(X, y, embeddings, batch_size=64, epochs=50, learning_rate=0.001, 
               patience=10, weight_decay=1e-5, hidden_dim=128, dropout_rate=0.3, 
               cv_folds=5):
    """Train and evaluate the neural network model"""
    print("Training neural network model...")
    
    # Split data
    X_train_all, X_test, y_train_all, y_test, emb_train_all, emb_test = train_test_split(
        X.values, y.values, embeddings, test_size=0.2, random_state=42
    )
    
    # Cross-validation setup
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_all)):
        print(f"\nTraining fold {fold+1}/{cv_folds}")
        
        # Split train and validation data
        X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]
        emb_train, emb_val = emb_train_all[train_idx], emb_train_all[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = QuestionDataset(X_train, emb_train, y_train)
        val_dataset = QuestionDataset(X_val, emb_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = NeuralDifficultyModel(
            features_dim=X_train.shape[1], 
            embedding_dim=emb_train.shape[1],
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Early stopping tracking
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for features, embeddings, targets in train_loader:
                features, embeddings, targets = features.to(device), embeddings.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(features, embeddings)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, embeddings, targets in val_loader:
                    features, embeddings, targets = features.to(device), embeddings.to(device), targets.to(device)
                    outputs = model(features, embeddings)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_rmse = np.sqrt(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f'neural_model_filtered_fold{fold+1}.pt')
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model for this fold
        model.load_state_dict(torch.load(f'neural_model_filtered_fold{fold+1}.pt'))
        
        # Calculate final validation RMSE for this fold
        model.eval()
        with torch.no_grad():
            val_predictions = []
            for features, embeddings, targets in val_loader:
                features, embeddings = features.to(device), embeddings.to(device)
                outputs = model(features, embeddings)
                val_predictions.extend(outputs.cpu().numpy())
            
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            cv_scores.append(val_rmse)
            print(f"Fold {fold+1} final validation RMSE: {val_rmse:.4f}")
    
    # Print cross-validation results
    print(f"\nCross-validation RMSE scores: {cv_scores}")
    print(f"Mean CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all training data
    print("\nTraining final model on all training data...")
    final_train_dataset = QuestionDataset(X_train_all, emb_train_all, y_train_all)
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
    
    final_model = NeuralDifficultyModel(
        features_dim=X_train_all.shape[1], 
        embedding_dim=emb_train_all.shape[1],
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    best_train_loss = float('inf')
    for epoch in range(epochs):
        # Training
        final_model.train()
        train_loss = 0
        for features, embeddings, targets in final_train_loader:
            features, embeddings, targets = features.to(device), embeddings.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(features, embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(final_train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(final_model.state_dict(), 'neural_model_filtered_final.pt')
    
    # Load best final model
    final_model.load_state_dict(torch.load('neural_model_filtered_final.pt'))
    
    # Evaluate on test set
    test_dataset = QuestionDataset(X_test, emb_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    final_model.eval()
    test_predictions = []
    actual_values = []
    with torch.no_grad():
        for features, embeddings, targets in test_loader:
            features, embeddings = features.to(device), embeddings.to(device)
            outputs = final_model(features, embeddings)
            test_predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.numpy())
    
    # Calculate test metrics
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create a plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, test_predictions, alpha=0.5)
    plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('Neural Network (Filtered): Actual vs Predicted Difficulty')
    plt.savefig('neural_prediction_scatter_filtered.png')
    
    # Save the trained model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'features_dim': X_train_all.shape[1],
        'embedding_dim': emb_train_all.shape[1],
        'hidden_dim': hidden_dim,
        'dropout_rate': dropout_rate,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'cv_scores': cv_scores
    }, 'neural_question_model_filtered.pt')
    
    return final_model, rmse, mae, r2, np.mean(cv_scores)

def main():
    # Load the data with filtering of very easy questions
    df = load_data(difficulty_threshold=-6)
    
    print("\nSample of the loaded data:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())
    
    # Define target variable as difficulty (continuous)
    target_col = 'irt_difficulty'
    if target_col in df.columns:
        y = df[target_col]
        
        print(f"\nTarget variable statistics after filtering:")
        print(f"Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        
        # Process features
        X_structured, X_embeddings = preprocess_features(df)
        
        # Set model hyperparameters
        model_params = {
            'batch_size': 64,
            'epochs': 50,
            'learning_rate': 0.001,
            'patience': 7,
            'weight_decay': 1e-5,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'cv_folds': 5
        }
        
        # Train and evaluate model
        model, rmse, mae, r2, cv_rmse = train_model(
            X_structured, y, X_embeddings, **model_params
        )
        
        # Write performance metrics to file
        with open('neural_model_filtered_metrics.txt', 'w') as f:
            f.write(f"## Neural Network Model with Embeddings (Filtered: difficulty >= -6)\n")
            f.write(f"- Mean CV RMSE: {cv_rmse:.4f}\n")
            f.write(f"- Test RMSE: {rmse:.4f}\n")
            f.write(f"- Test MAE: {mae:.4f}\n")
            f.write(f"- Test R²: {r2:.4f}\n")
            f.write(f"\nModel Hyperparameters:\n")
            for param, value in model_params.items():
                f.write(f"- {param}: {value}\n")
        
        # Update the comparison metrics file
        try:
            with open('metrics_comparison.txt', 'a') as f:
                f.write(f"\n## Neural Network Model with all-MiniLM-L6-v2 embeddings (Filtered: difficulty >= -6)\n")
                f.write(f"- Mean CV RMSE: {cv_rmse:.4f}\n")
                f.write(f"- Test RMSE: {rmse:.4f}\n")
                f.write(f"- Test MAE: {mae:.4f}\n")
                f.write(f"- Test R²: {r2:.4f}\n")
        except:
            print("Could not update metrics_comparison.txt")
        
        print("\nNeural model training completed. Files saved:")
        print("- neural_question_model_filtered.pt: The trained model")
        print("- feature_processors_neural_filtered.pkl: Feature processing objects")
        print("- neural_prediction_scatter_filtered.png: Scatter plot of actual vs predicted values")
        print("- neural_model_filtered_metrics.txt: Metrics summary")
        
    else:
        print(f"Error: Target column '{target_col}' not found in data.")
        print(f"Available columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main() 
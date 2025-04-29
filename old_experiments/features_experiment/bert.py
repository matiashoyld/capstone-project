#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT Regression Model for Question Difficulty Prediction

This script implements a regression model based on the BERT base uncased model
to predict question difficulty. It uses the question text and options as input
and predicts a continuous difficulty score.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import logging
import requests
from requests.exceptions import HTTPError, ConnectionError
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256  # Reduced from 512 to prevent memory issues

def combine_question_options(row):
    """
    Combine question text and options into a single text string
    
    Args:
        row: DataFrame row containing question and options
        
    Returns:
        String containing the question and options
    """
    question = row['question_title'] if 'question_title' in row else ""
    
    # Combine options
    options_text = ""
    if 'option_a' in row and not pd.isna(row['option_a']):
        options_text += f" A) {row['option_a']}"
    if 'option_b' in row and not pd.isna(row['option_b']):
        options_text += f" B) {row['option_b']}"
    if 'option_c' in row and not pd.isna(row['option_c']):
        options_text += f" C) {row['option_c']}"
    if 'option_d' in row and not pd.isna(row['option_d']):
        options_text += f" D) {row['option_d']}"
    if 'option_e' in row and not pd.isna(row['option_e']):
        options_text += f" E) {row['option_e']}"
    
    return f"Question: {question} Options: {options_text}"

def check_model_exists_on_hub(hub_repo_id, hub_token=None):
    """
    Check if the model exists on the Hugging Face Hub
    
    Args:
        hub_repo_id: Repository ID on Hugging Face Hub
        hub_token: Hugging Face API token (optional)
        
    Returns:
        Boolean indicating if the model exists
    """
    api = HfApi(token=hub_token)
    try:
        # Try to get the model info
        api.model_info(hub_repo_id)
        return True
    except (HTTPError, ConnectionError):
        return False

def tokenize_function(examples):
    """
    Tokenize text for BERT model (obsolete, kept for reference)
    
    Args:
        examples: Examples to tokenize
        
    Returns:
        Tokenized examples
    """
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    # Include labels for training
    if "irt_difficulty" in examples:
        tokenized["labels"] = examples["irt_difficulty"]
    
    return tokenized

def load_data(file_path, difficulty_threshold=None):
    """
    Load and preprocess question data
    
    Args:
        file_path: Path to the CSV file containing question data
        difficulty_threshold: Optional threshold to filter questions by difficulty
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load the question data
    df = pd.read_csv(file_path)
    
    # Check if IRT difficulty data needs to be merged
    if 'irt_difficulty' not in df.columns:
        irt_file_path = os.path.join(os.path.dirname(file_path), "question_difficulties_irt.csv")
        if os.path.exists(irt_file_path):
            logger.info(f"Merging IRT difficulties from {irt_file_path}")
            irt_df = pd.read_csv(irt_file_path)
            df = df.merge(irt_df[['question_id', 'irt_difficulty']], on='question_id', how='left')
    
    # Filter questions by difficulty if threshold is provided
    if difficulty_threshold is not None and 'irt_difficulty' in df.columns:
        original_count = len(df)
        df = df[df['irt_difficulty'] >= difficulty_threshold]
        filtered_count = original_count - len(df)
        logger.info(f"Filtered out {filtered_count} questions with difficulty < {difficulty_threshold}")
        logger.info(f"Remaining questions: {len(df)}")
    
    # Create text field by combining question and options
    df['text'] = df.apply(combine_question_options, axis=1)
    
    # Remove questions with missing target variable
    if 'irt_difficulty' in df.columns:
        df = df.dropna(subset=['irt_difficulty'])
    
    return df

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for regression
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def plot_predictions(y_true, y_pred, file_path="prediction_scatter_bert.png"):
    """
    Create a scatter plot of true vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        file_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.title('BERT Model: Actual vs Predicted Difficulty')
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Prediction scatter plot saved to {file_path}")

def save_metrics(metrics, file_path="bert_model_metrics.txt"):
    """
    Save evaluation metrics to a text file
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the metrics
    """
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    logger.info(f"Metrics saved to {file_path}")

def train_bert_model(
    data_file,
    output_dir="bert_model",
    difficulty_threshold=None,
    epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    test_size=0.2,
    random_state=42,
    push_to_hub=False,
    hub_repo_id=None,
    hub_token=None
):
    """
    Train a BERT model for question difficulty prediction
    
    Args:
        data_file: Path to the CSV file containing question data
        output_dir: Directory to save the model
        difficulty_threshold: Optional threshold to filter questions by difficulty
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        push_to_hub: Whether to push the model to Hugging Face Hub
        hub_repo_id: Repository ID for Hugging Face Hub
        hub_token: Hugging Face API token
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load and preprocess data
    df = load_data(data_file, difficulty_threshold)
    
    # Ensure the target variable is a column in the dataframe
    if 'irt_difficulty' not in df.columns:
        raise ValueError("Target variable 'irt_difficulty' not found in the data")
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize datasets
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Add labels
        result["labels"] = examples["irt_difficulty"]
        return result
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    # Load BERT model for regression (sequence classification with 1 label)
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=1,
        problem_type="regression"
    )
    
    # Define training arguments
    if push_to_hub and hub_repo_id and not check_model_exists_on_hub(hub_repo_id, hub_token):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            push_to_hub=True,
            hub_model_id=hub_repo_id,
            hub_token=hub_token,
            report_to="none",  # Disable wandb logging
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            report_to="none",  # Disable wandb logging
        )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting model training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Save the best model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # Create prediction plot
    test_preds = trainer.predict(test_dataset)
    y_true = test_df['irt_difficulty'].values
    y_pred = test_preds.predictions.squeeze()
    plot_predictions(y_true, y_pred)
    
    # Save metrics
    test_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    save_metrics(test_metrics)
    
    # Update metrics comparison file
    update_metrics_comparison(test_metrics, difficulty_threshold)
    
    return test_metrics

def predict_difficulty(question_text, options, model_path="bert_model"):
    """
    Predict the difficulty of a question
    
    Args:
        question_text: The question text
        options: Dictionary of options (e.g., {"A": "...", "B": "..."})
        model_path: Path to the trained model
        
    Returns:
        Predicted difficulty score
    """
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Format the input text
    options_text = ""
    for key, value in options.items():
        options_text += f" {key}) {value}"
    
    input_text = f"Question: {question_text} Options: {options_text}"
    
    # Tokenize the input
    inputs = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze().item()
    
    return predictions

def update_metrics_comparison(metrics, difficulty_threshold=None):
    """
    Update the metrics comparison file with BERT model results
    
    Args:
        metrics: Dictionary of metrics
        difficulty_threshold: The difficulty threshold used for filtering
    """
    try:
        # Try to read existing metrics comparison file
        comparison_file = "metrics_comparison.txt"
        
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                content = f.read()
        else:
            content = "# Model Performance Metrics Comparison\n\n"
        
        # Create entry for BERT model
        if difficulty_threshold is not None:
            model_name = f"BERT base uncased (Filtered: difficulty >= {difficulty_threshold})"
            note = f"- Note: This model excludes questions with difficulty < {difficulty_threshold}"
        else:
            model_name = "BERT base uncased"
            note = ""
        
        # Create new entry
        new_entry = (
            f"\n### {model_name}\n"
            f"- Test RMSE: {metrics['rmse']:.4f}\n"
            f"- Test MAE: {metrics['mae']:.4f}\n"
            f"- Test R²: {metrics['r2']:.4f}\n"
            f"{note}\n"
        )
        
        # Find the appropriate section or add a new one
        bert_section = content.find("## BERT Models")
        if bert_section == -1:
            # Add new section for BERT models
            content += "\n## BERT Models\n"
            content += new_entry
        else:
            # Find the end of the BERT section
            next_section = content.find("##", bert_section + 1)
            if next_section == -1:
                # BERT section is the last section
                content += new_entry
            else:
                # Insert before the next section
                content = content[:next_section] + new_entry + content[next_section:]
        
        # Write updated content
        with open(comparison_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated metrics comparison file with BERT model results.")
        
    except Exception as e:
        logger.error(f"Error updating metrics comparison: {str(e)}")

def main():
    """
    Main function to run the BERT model training
    """
    # Parse command line arguments (in a full implementation)
    # For now, use default values
    data_file = "questions_master.csv"
    output_dir = "bert_model"
    difficulty_threshold = -6  # Filter out very easy questions (same as in other models)
    
    # Train the model
    metrics = train_bert_model(
        data_file=data_file,
        output_dir=output_dir,
        difficulty_threshold=difficulty_threshold,
        epochs=3,  # Reduced epochs
        batch_size=4,  # Smaller batch size
        learning_rate=2e-5
    )
    
    logger.info("BERT model training completed.")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {metrics['mae']:.4f}")
    logger.info(f"Test R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main() 
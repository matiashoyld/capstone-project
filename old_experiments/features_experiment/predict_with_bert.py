#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT Question Difficulty Prediction

This script loads a trained BERT model and predicts the difficulty of new questions
based on their text and options, without requiring response data.
"""

import os
import sys
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
MODEL_PATH = "bert_model"

def combine_question_options(question_text, options):
    """
    Combine question text and options into a single text string
    
    Args:
        question_text: The question text
        options: Dictionary of options (e.g., {"A": "option text", "B": "option text"})
        
    Returns:
        String containing the question and options
    """
    options_text = ""
    for key, value in sorted(options.items()):
        options_text += f" {key}) {value}"
    
    return f"Question: {question_text} Options: {options_text}"

def load_model(model_path=MODEL_PATH):
    """
    Load the trained BERT model
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Loaded model and tokenizer
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path '{model_path}' does not exist!")
        sys.exit(1)
    
    try:
        logger.info(f"Loading model from {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        sys.exit(1)

def predict_difficulty(question_text, options, model, tokenizer):
    """
    Predict the difficulty of a question
    
    Args:
        question_text: The question text
        options: Dictionary of options (e.g., {"A": "...", "B": "..."})
        model: Loaded BERT model
        tokenizer: BERT tokenizer
        
    Returns:
        Predicted difficulty score
    """
    # Format the input text
    input_text = combine_question_options(question_text, options)
    
    # Tokenize the input
    inputs = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.squeeze().item()
    
    return prediction

def predict_batch_from_csv(csv_file, model_path=MODEL_PATH, output_file=None):
    """
    Predict difficulties for a batch of questions from a CSV file
    
    Args:
        csv_file: Path to the CSV file containing questions
        model_path: Path to the trained model
        output_file: Path to save the predictions (optional)
        
    Returns:
        DataFrame with original data and predictions
    """
    if not os.path.exists(csv_file):
        logger.error(f"CSV file '{csv_file}' does not exist!")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load questions
    logger.info(f"Loading questions from {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        sys.exit(1)
    
    # Check required columns
    required_cols = ['question_title']
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    
    if 'question_title' not in df.columns:
        logger.error(f"CSV file must contain 'question_title' column!")
        sys.exit(1)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    
    for idx, row in df.iterrows():
        # Get question text
        question_text = row['question_title']
        
        # Get options
        options = {}
        for i, col in enumerate(option_cols):
            if col in df.columns and not pd.isna(row[col]):
                options[chr(65 + i)] = row[col]  # A, B, C, etc.
        
        # Predict difficulty
        try:
            difficulty = predict_difficulty(question_text, options, model, tokenizer)
            predictions.append(difficulty)
        except Exception as e:
            logger.error(f"Error predicting difficulty for question {idx}: {str(e)}")
            predictions.append(None)
    
    # Add predictions to DataFrame
    df['predicted_difficulty'] = predictions
    
    # Save predictions if output file is specified
    if output_file:
        logger.info(f"Saving predictions to {output_file}")
        df.to_csv(output_file, index=False)
    
    return df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict question difficulty using BERT')
    
    # Add arguments
    parser.add_argument('--input', type=str, help='Input CSV file with questions')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--question', type=str, help='Single question text to predict')
    parser.add_argument('--options', type=str, nargs='+', help='Options for the question (A B C D)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    if args.input:
        # Batch prediction from CSV
        predict_batch_from_csv(args.input, args.model, args.output)
    elif args.question:
        # Single question prediction
        if not args.options or len(args.options) < 2:
            logger.error("Please provide at least two options for the question!")
            sys.exit(1)
        
        # Load model
        model, tokenizer = load_model(args.model)
        
        # Create options dictionary
        options = {}
        for i, option_text in enumerate(args.options):
            options[chr(65 + i)] = option_text  # A, B, C, etc.
        
        # Predict difficulty
        difficulty = predict_difficulty(args.question, options, model, tokenizer)
        print(f"Predicted difficulty: {difficulty:.4f}")
    else:
        logger.error("Please provide either an input CSV file or a question and options!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
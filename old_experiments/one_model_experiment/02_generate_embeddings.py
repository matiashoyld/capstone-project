import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to get sentence embedding.
    
    Args:
        model_output: Output from the model
        attention_mask: Attention mask from tokenizer
        
    Returns:
        Mean pooled embeddings
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def get_embeddings(texts, tokenizer, model, device, batch_size=16):
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        tokenizer: Tokenizer for text preprocessing
        model: Model to generate embeddings
        device: Device to run the model on
        batch_size: Batch size for processing
        
    Returns:
        Normalized embeddings for each text
    """
    all_embeddings = []
    
    # Process in batches to avoid OOM errors
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        # Get model output
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean pooling
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        
        all_embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings.numpy()

def format_question_and_answers(row):
    """
    Format the question and answers according to the specified format.
    
    Args:
        row: DataFrame row containing question and options
        
    Returns:
        Formatted text
    """
    # Determine correct and wrong answers
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    options = [row[col] for col in option_cols]
    
    # Map correct option letter to index
    correct_letter = row['correct_option_letter']
    if correct_letter in ['A', 'B', 'C', 'D', 'E']:
        correct_idx = ord(correct_letter) - ord('A')
        if correct_idx < len(option_cols):
            correct_answer = options[correct_idx]
            wrong_answers = [opt for i, opt in enumerate(options) if i != correct_idx]
            
            # Some questions may have fewer than 5 options, so filter out missing ones
            wrong_answers = [ans for ans in wrong_answers if pd.notna(ans)]
            
            # Format with question, correct answer, and wrong answers
            formatted_text = f"Question: {row['question_title']}\n"
            formatted_text += f"Correct Answer: {correct_answer}\n"
            
            # Add wrong answers (up to 4)
            for i, wrong_ans in enumerate(wrong_answers[:4]):
                formatted_text += f"Wrong Answer {i+1}: {wrong_ans}\n"
                
            return formatted_text
    
    # Default return if correct answer couldn't be determined
    return f"Question: {row['question_title']}"

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the dataset from the data directory
    data_path = os.path.join(current_dir, 'data', 'merged_features_filtered.csv')
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Keep only unique questions (no duplicates)
    question_df = df.drop_duplicates(subset=['question_id'])
    print(f"Found {len(question_df)} unique questions")
    
    # Load model and tokenizer
    print("Loading ModernBERT model and tokenizer...")
    model_name = "nomic-ai/modernbert-embed-base"  # Using Nomic AI's embedding-optimized version of ModernBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Format questions and answers
    print("Formatting questions and answers...")
    question_df['formatted_text'] = question_df.apply(format_question_and_answers, axis=1)
    
    # Generate embeddings for formatted text
    print("Generating embeddings for formatted questions and answers...")
    formatted_embeddings = get_embeddings(
        question_df['formatted_text'].tolist(),
        tokenizer,
        model,
        device
    )
    
    # Generate embeddings for individual options
    print("Generating embeddings for individual options...")
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    option_embeddings = {}
    
    for col in option_cols:
        # Filter out NaN values
        valid_options = question_df[col].dropna().tolist()
        if valid_options:
            print(f"Generating embeddings for {col}...")
            embeddings = get_embeddings(valid_options, tokenizer, model, device)
            option_embeddings[col] = embeddings
    
    # Create a dictionary to store all embeddings
    embeddings_dict = {
        'question_ids': question_df['question_id'].tolist(),
        'formatted_embeddings': formatted_embeddings,
        'option_embeddings': option_embeddings
    }
    
    # Save embeddings to pickle file
    output_path = os.path.join(current_dir, 'data', 'question_embeddings.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Embeddings saved to {output_path}")
    print(f"Formatted embeddings shape: {formatted_embeddings.shape}")
    for col, embs in option_embeddings.items():
        print(f"{col} embeddings shape: {embs.shape}")

if __name__ == "__main__":
    main() 
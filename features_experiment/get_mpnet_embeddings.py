# Generate embeddings using the all-mpnet-base-v2 model
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import torch

print("Loading the all-mpnet-base-v2 model...")
# Check if MPS (Apple Silicon GPU) is available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
print("Model loaded successfully.")

# Load questions data
print("Loading questions data...")
questions = pd.read_csv('questions_filtered.csv')
print(f"Loaded {len(questions)} questions.")

# Function to format question with correct and wrong answers (similar to ModernBERT approach)
def format_question_with_answers(row):
    correct_option = row['correct_option_letter'].upper()
    options = {
        'A': row['option_a'] if pd.notna(row['option_a']) else None,
        'B': row['option_b'] if pd.notna(row['option_b']) else None,
        'C': row['option_c'] if pd.notna(row['option_c']) else None,
        'D': row['option_d'] if pd.notna(row['option_d']) else None,
        'E': row['option_e'] if pd.notna(row['option_e']) else None
    }
    
    # Get the correct answer text
    correct_answer_text = options[correct_option]
    
    # Remove the correct option from options dict
    options.pop(correct_option)
    
    # Get wrong answers (only non-None values)
    wrong_answers = [opt for key, opt in options.items() if opt is not None]
    
    # Format the question as specified
    text = f"Question: {row['question_title']}\n"
    text += f"Correct Answer: {correct_answer_text}\n"
    
    # Add wrong answers
    for i, wrong_answer in enumerate(wrong_answers, 1):
        text += f"Wrong Answer {i}: {wrong_answer}\n"
    
    return text

# Original concatenate function (keep for backward compatibility)
def concatenate_text(row):
    text = f"Question: {row['question_title']}\n"
    
    option_letters = ['A', 'B', 'C', 'D', 'E']
    option_columns = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    
    for letter, option in zip(option_letters, option_columns):
        if option in row and pd.notna(row[option]):
            text += f"{letter}) {str(row[option])}\n"
    
    return text

# Generate embeddings using MPNet
def get_mpnet_embeddings(texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# Create columns with different text formats
print("Preparing question texts...")
questions['formatted_text'] = questions.apply(format_question_with_answers, axis=1)
questions['concatenated_text'] = questions.apply(concatenate_text, axis=1)

# Generate embeddings for formatted text (similar to ModernBERT approach)
print("Generating MPNet embeddings for formatted questions... (this may take a while)")
start_time = time.time()
formatted_embeddings = get_mpnet_embeddings(questions['formatted_text'].tolist())
end_time = time.time()

print(f"Formatted embedding generation completed in {end_time - start_time:.2f} seconds.")
print(f"Embedding dimension: {formatted_embeddings.shape[1]}")  # Should be 768 for mpnet

# Generate embeddings for original concatenated text (backward compatibility)
print("Generating MPNet embeddings for concatenated questions...")
start_time = time.time()
concatenated_embeddings = get_mpnet_embeddings(questions['concatenated_text'].tolist())
end_time = time.time()

print(f"Concatenated embedding generation completed in {end_time - start_time:.2f} seconds.")

# Generate embeddings for individual answer options
print("Generating embeddings for individual answer options...")
start_time = time.time()

option_columns = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
option_embeddings = {}

for option_col in option_columns:
    # Filter out None values and convert to list
    option_texts = questions[option_col].dropna().tolist()
    if option_texts:
        print(f"Generating embeddings for {option_col}...")
        embeddings = get_mpnet_embeddings(option_texts)
        option_embeddings[option_col] = embeddings
        print(f"Completed {option_col} embeddings with shape: {embeddings.shape}")

# Create columns for individual option embeddings
for option_col in option_columns:
    questions[f'{option_col}_embedding'] = None
    # Only set values for rows where the option exists
    mask = questions[option_col].notna()
    if mask.any():
        # Get indices of non-NA values
        valid_indices = questions[mask].index
        # Map embeddings to these indices
        embedding_dict = {idx: emb for idx, emb in zip(valid_indices, option_embeddings[option_col])}
        # Update the DataFrame
        questions.loc[mask, f'{option_col}_embedding'] = questions.loc[mask].index.map(embedding_dict)

end_time = time.time()
print(f"Option embeddings generation completed in {end_time - start_time:.2f} seconds.")

# Create a DataFrame with question_id, formatted embedding, concatenated embedding, and individual option embeddings
questions_embeddings = pd.DataFrame({
    'question_id': questions['question_id'],
    'formatted_embedding': list(formatted_embeddings),
    'concatenated_embedding': list(concatenated_embeddings)
})

# Add individual option embeddings to the DataFrame
for option_col in option_columns:
    questions_embeddings[f'{option_col}_embedding'] = questions[f'{option_col}_embedding']

# Display a sample of the DataFrame
print("\nSample of questions_embeddings DataFrame:")
print(questions_embeddings.head())

# Save embeddings to a pickle file
print("Saving embeddings to pickle file...")
questions_embeddings.to_pickle('questions_mpnet_embeddings.pkl')

print("Done! Saved to questions_mpnet_embeddings.pkl") 
# %%
# 1) Imports and Environment Setup

from dotenv import load_dotenv  # For loading environment variables from a .env file
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
import logging
import requests
from requests.exceptions import HTTPError, ConnectionError
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

print("[INFO] Environment variables loaded and libraries imported.")

# %%
questions = pd.read_csv("questions_with_distractors.csv")
questions_images = pd.read_csv("questions_with_images.csv")
# Load the correct difficulty values from question_difficulties_irt.csv
question_difficulties = pd.read_csv("question_difficulties_irt.csv")

# %%
# Add a column to questions df indicating if the question has an image
questions_images_list = questions_images['question_id'].tolist()
questions['has_image'] = questions['question_id'].isin(questions_images_list)

# Merge the questions dataframe with the correct difficulty values
questions = questions.merge(question_difficulties[['question_id', 'irt_difficulty']], 
                           on='question_id', 
                           how='left')

# Log how many questions have difficulty values
print(f"Questions with IRT difficulty values: {questions['irt_difficulty'].notna().sum()} out of {len(questions)}")

# Display the first few rows to verify
print(f"Total questions: {len(questions)}")
print(f"Questions with images: {questions['has_image'].sum()}")
print("\nSample of questions with has_image column:")
print(questions[['question_id', 'has_image', 'irt_difficulty']].head())

# %%
# Add a total_count column that sums all the individual option counts
questions['total_count'] = questions['count_a'] + questions['count_b'] + questions['count_c'] + questions['count_d'] + questions['count_e']

# Display the first few rows to verify
print("\nSample of questions with total_count column:")
print(questions[['question_id', 'count_a', 'count_b', 'count_c', 'count_d', 'count_e', 'total_count', 'irt_difficulty']].head())

# %%
# Filter out questions with less than 10 total responses
filtered_questions = questions[questions['total_count'] >= 10]

# Display information about the filtering
print(f"Original number of questions: {len(questions)}")
print(f"Number of questions with at least 10 responses: {len(filtered_questions)}")
print(f"Number of questions removed: {len(questions) - len(filtered_questions)}")

# Replace the original dataframe with the filtered one
questions = filtered_questions

# Display the first few rows to verify
print("\nSample of filtered questions:")
print(questions[['question_id', 'total_count']].head())

# %%
questions.to_csv("questions_filtered.csv", index=False)

# %%
# Fine-tune ModernBERT to predict question difficulty
print("[INFO] Starting ModernBERT fine-tuning setup...")

# Check for MPS (Apple Silicon GPU) availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# %%
# Prepare the data for training
# We'll use question_title, option_a, option_b, option_c, option_d, option_e as input features
# and difficulty as the target

# Function to combine question and options into a single text
def combine_question_options(row):
    text = f"Topic: {row.get('topic_name', '')}\n"
    text += f"Subject: {row.get('subject_name', '')}\n"
    text += f"Question: {row['question_title']}\n"
    text += f"A: {row['option_a']}\n"
    text += f"B: {row['option_b']}\n"
    text += f"C: {row['option_c']}\n"
    text += f"D: {row['option_d']}\n"
    
    # Add option E if it's not empty or NaN
    if pd.notna(row['option_e']) and row['option_e'].strip():
        text += f"E: {row['option_e']}"
    
    return text

# Create input texts
questions['input_text'] = questions.apply(combine_question_options, axis=1)

# Split the data into train and test sets (80/20)
train_df, test_df = train_test_split(
    questions, 
    test_size=0.2, 
    random_state=42
)

print(f"[INFO] Train set size: {len(train_df)}, Test set size: {len(test_df)}")

# %%
# Load ModernBERT tokenizer
model_id = "answerdotai/ModernBERT-base"

# Helper function to check if ModernBERT is available
def check_model_exists_on_hub(hub_repo_id, hub_token=None):
    """Check if a model repository exists on the Hugging Face Hub."""
    headers = {"Authorization": f"Bearer {hub_token}"} if hub_token else None
    
    try:
        response = requests.head(
            f"https://huggingface.co/api/models/{hub_repo_id}",
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except (HTTPError, ConnectionError, requests.Timeout) as e:
        print(f"Error checking if model exists on Hub: {str(e)}")
        return False

print(f"[INFO] Checking if {model_id} is available on Hugging Face Hub...")
if check_model_exists_on_hub(model_id):
    print(f"[INFO] {model_id} is available on Hugging Face Hub.")
else:
    print(f"[WARNING] {model_id} may not be available on Hugging Face Hub.")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"[INFO] Successfully loaded tokenizer from {model_id}")
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer: {str(e)}")
    # Fall back to another BERT model as tokenizer
    print("[INFO] Falling back to bert-base-uncased tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['input_text', 'irt_difficulty']])
test_dataset = Dataset.from_pandas(test_df[['input_text', 'irt_difficulty']])

# Tokenize function - with batching for efficiency
def tokenize_function(examples):
    return tokenizer(
        examples['input_text'], 
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Apply tokenization
print("[INFO] Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename label column to match what the Trainer expects
train_dataset = train_dataset.rename_column('irt_difficulty', 'labels')
test_dataset = test_dataset.rename_column('irt_difficulty', 'labels')

# Format datasets to return PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# %%
# Initialize the model for regression
print("[INFO] Loading ModernBERT model...")

try:
    # Set environment variable for HF transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # First try to load the model config to see if that works
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, num_labels=1, problem_type="regression")
    print(f"[INFO] Successfully loaded config from {model_id}")
    
    # Then load the actual model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config
    )
    print(f"[INFO] Successfully loaded model from {model_id}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {str(e)}")
    raise RuntimeError(f"Could not load ModernBERT. Error: {str(e)}")

# Define custom metrics for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

# Calculate steps for learning rate scheduler
batch_size = 4
num_train_examples = len(train_dataset)
steps_per_epoch = (num_train_examples + batch_size - 1) // batch_size
total_steps = steps_per_epoch * 5  # 5 epochs
warmup_steps = int(0.1 * total_steps)  # 10% of total steps

# Define training arguments
model_output_dir = "modernbert-question-irt-difficulty"
print(f"[INFO] Setting up training arguments. Output dir: {model_output_dir}")

training_args = TrainingArguments(
    output_dir=model_output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,  # Lower learning rate for stability
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=steps_per_epoch // 5,  # Log ~5 times per epoch
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    greater_is_better=False,  # Lower RMSE is better
    use_mps_device=torch.backends.mps.is_available(),  # For Mac M2
    fp16=False,  # Disable mixed precision (not needed for MPS)
    save_total_limit=2,
    weight_decay=0.01,  # Add L2 regularization
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine_with_restarts"  # Better than linear
)

# Create the trainer
print("[INFO] Creating Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# %%
# Train the model
print("[INFO] Starting model training...")
try:
    train_result = trainer.train()
    print("[INFO] Training completed successfully!")
    
    # Save model and training metrics
    trainer.save_model(model_output_dir)
    trainer.save_state()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
except Exception as e:
    print(f"[ERROR] Training failed: {str(e)}")
    raise

# %%
# Evaluate on test set
print("[INFO] Evaluating model on test set...")
results = trainer.evaluate()
print(f"Test results: {results}")

# %%
# Save the model
model_save_path = "modernbert-question-irt-difficulty-final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"[INFO] Model saved to {model_save_path}")

# %%
# Example of making a prediction with the fine-tuned model
def predict_difficulty(question_text, options, model_path):
    # Load the model and tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Combine question and options
    text = f"Question: {question_text}\n"
    for i, option in enumerate(options):
        text += f"{chr(65+i)}: {option}\n"
    
    # Tokenize
    inputs = loaded_tokenizer(
        text, 
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device if available
    if torch.backends.mps.is_available():
        inputs = {k: v.to('mps') for k, v in inputs.items()}
        loaded_model = loaded_model.to('mps')
    
    # Get prediction
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predicted_difficulty = outputs.logits.item()
    
    return predicted_difficulty

# Test with a sample question
if len(test_df) > 0:
    sample = test_df.iloc[0]
    sample_question = sample['question_title']
    sample_options = [
        sample['option_a'],
        sample['option_b'],
        sample['option_c'],
        sample['option_d']
    ]
    if pd.notna(sample['option_e']) and sample['option_e'].strip():
        sample_options.append(sample['option_e'])
    
    print("\n[INFO] Example prediction:")
    print(f"Question: {sample_question}")
    for i, option in enumerate(sample_options):
        print(f"{chr(65+i)}: {option}")
    
    actual_difficulty = sample['irt_difficulty']
    predicted_difficulty = predict_difficulty(
        sample_question, 
        sample_options, 
        model_save_path
    )
    
    print(f"Actual difficulty: {actual_difficulty}")
    print(f"Predicted difficulty: {predicted_difficulty}")
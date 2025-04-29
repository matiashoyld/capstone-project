import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
from tqdm import tqdm
import pickle
import os

logger = logging.getLogger(__name__)

def _mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings using the attention mask."""
    token_embeddings = model_output[0]  # First element of model_output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def _get_batch_embeddings(texts: list[str], tokenizer, model, device):
    """Generates embeddings for a single batch of texts."""
    if not texts:
        return np.array([])
        
    # Handle potential non-string data gracefully
    processed_texts = [str(text) if pd.notna(text) else "" for text in texts] 

    encoded_input = tokenizer(
        processed_texts, 
        padding=True, 
        truncation=True, 
        max_length=512, # Standard max length for many BERT-like models
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    batch_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1) # L2 Normalization
    return batch_embeddings.cpu().numpy()

def generate_text_embeddings(
    data_df: pd.DataFrame,
    text_col: str,
    id_col: str,
    model_name: str = "nomic-ai/modernbert-embed-base", # Changed default model
    batch_size: int = 32,
    device: str | None = None
) -> dict[any, np.ndarray]:
    """
    Generates embeddings for a text column in a DataFrame using a transformer model.

    Args:
        data_df: DataFrame containing the text data and identifiers.
        text_col: The name of the column containing the text to embed.
        id_col: The name of the column containing the unique identifiers (e.g., 'question_id').
        model_name: The name of the Hugging Face transformer model to use.
        batch_size: How many texts to process at once.
        device: The device to run on ('cuda', 'mps', 'cpu'). Auto-detected if None.

    Returns:
        A dictionary mapping identifiers from id_col to their corresponding embedding arrays.
        Returns an empty dictionary if required columns are missing or an error occurs.
    """
    logger.info(f"Starting embedding generation with model '{model_name}'.")

    if text_col not in data_df.columns:
        logger.error(f"Text column '{text_col}' not found in DataFrame.")
        return {}
    if id_col not in data_df.columns:
        logger.error(f"ID column '{id_col}' not found in DataFrame.")
        return {}

    # --- Device Setup ---
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    torch_device = torch.device(device)

    # --- Load Model and Tokenizer ---
    try:
        # Trust remote code for certain models like nomic-embed-text
        trust_remote = 'nomic' in model_name 
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote).to(torch_device)
        model.eval() # Set model to evaluation mode
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer '{model_name}': {e}")
        return {}

    # --- Prepare Data ---
    ids = data_df[id_col].tolist()
    texts_to_embed = data_df[text_col].tolist()
    embeddings_dict = {}

    # --- Process in Batches ---
    logger.info(f"Generating embeddings for {len(texts_to_embed)} texts in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings"):
        batch_texts = texts_to_embed[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        try:
            batch_embeddings = _get_batch_embeddings(batch_texts, tokenizer, model, torch_device)
            
            # Store embeddings in the dictionary
            for idx, embedding in enumerate(batch_embeddings):
                embeddings_dict[batch_ids[idx]] = embedding
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")
            # Optionally skip the batch or handle specific errors
            continue

    logger.info(f"Generated embeddings for {len(embeddings_dict)} items.")
    return embeddings_dict
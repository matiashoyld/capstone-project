"""Lightweight evaluation utilities."""

import os, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from modules.irt import estimate_irt_1pl_difficulty, estimate_irt_2pl_params

# ------------------------------------------------------------------
#  Build dataset for a given set of question IDs
# ------------------------------------------------------------------

def make_dataset(df, q_ids, preprocessors, combined_embeddings,
                 user_col, question_col, correctness_col):
    scaler   = preprocessors["scaler"]
    uid2idx  = preprocessors["user_id_to_index"]
    num_cols = preprocessors["numerical_features"]
    emb_dim  = preprocessors["embedding_dim"]

    sub = df[df[question_col].isin(q_ids)].copy()

    X_user = sub[user_col].map(lambda u: uid2idx.get(u, 0)).values
    X_num  = scaler.transform(sub[num_cols])
    X_emb  = None
    if emb_dim > 0:
        emb_map = combined_embeddings["formatted_embeddings"]
        X_emb = np.vstack([emb_map.get(q, np.zeros(emb_dim)) for q in sub[question_col]])

    dataset = {
        "user_input": X_user,
        "numerical_input": X_num,
    }
    if emb_dim > 0:
        dataset["embedding_input"] = X_emb

    y = sub[correctness_col].values.astype(int)
    return dataset, y

# ------------------------------------------------------------------
#  Evaluate binary classifier
# ------------------------------------------------------------------

def evaluate_model(model, dataset, labels):
    probs = model.predict(dataset, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    return dict(
        accuracy  = float(accuracy_score(labels, preds)),
        auc       = float(roc_auc_score(labels, probs)),
        precision = float(precision_score(labels, preds, zero_division=0)),
        recall    = float(recall_score(labels, preds, zero_division=0)),
        f1        = float(f1_score(labels, preds, zero_division=0)),
    )

# ------------------------------------------------------------------
#  User × question prediction matrix
# ------------------------------------------------------------------

def prediction_matrix(data_df, q_ids, preprocessors, model, combined_embeddings,
                      user_col, question_col):
    uid2idx  = preprocessors["user_id_to_index"]
    scaler   = preprocessors["scaler"]
    num_cols = preprocessors["numerical_features"]
    emb_dim  = preprocessors["embedding_dim"]
    emb_map  = combined_embeddings["formatted_embeddings"]
    default_emb = np.zeros(emb_dim)

    # Ensure all users in uid2idx (except 0) are considered
    # Filter users if they are not present in the data_df
    valid_user_ids = data_df[user_col].unique()
    users = [u for u, idx in uid2idx.items() if idx != 0 and u in valid_user_ids]
    if not users:
        print("Warning: No valid users found for prediction matrix.")
        return pd.DataFrame() # Return empty DataFrame
    
    mat = []

    # --- Prepare inputs that are constant across users ---
    # Numerical features (ensure order matches q_ids)
    q_features_df = data_df[[question_col] + num_cols].drop_duplicates(subset=[question_col]).set_index(question_col)
    q_features_ordered = q_features_df.reindex(q_ids) # Ensure order and presence
    
    # Handle potential NaNs introduced by reindexing (e.g., if a q_id is not in data_df)
    # Fill with zeros or median/mean if appropriate - using zero for simplicity here
    q_features_ordered = q_features_ordered.fillna(0)
    
    num_batch_unscaled = q_features_ordered[num_cols].values
    num_batch_scaled = scaler.transform(num_batch_unscaled) # Shape: (len(q_ids), num_feature_count)

    # Embeddings (only if emb_dim > 0)
    emb_batch = None
    if emb_dim > 0:
        # Ensure emb_map exists and handle potential missing q_ids
        emb_batch = np.vstack([emb_map.get(q, default_emb) for q in q_ids]) # Shape: (len(q_ids), emb_dim)
    # --- End constant inputs ---

    for u in users:
        user_idx_arr = np.array([uid2idx[u]] * len(q_ids)) # Shape: (len(q_ids),)

        inputs = {
            "user_input": user_idx_arr,
            "numerical_input": num_batch_scaled, # Use pre-calculated batch
        }
        if emb_dim > 0 and emb_batch is not None:
            inputs["embedding_input"] = emb_batch # Use pre-calculated batch

        # Ensure model prediction handles potential shape mismatches if q_ids have issues
        try:
            probs = model.predict(inputs, verbose=0).flatten()
            # Ensure probs length matches q_ids length
            if len(probs) != len(q_ids):
                 print(f"Warning: Prediction length mismatch for user {u}. Expected {len(q_ids)}, got {len(probs)}. Skipping user.")
                 continue # Skip this user if prediction length is wrong
            mat.append(probs)
        except Exception as e:
            print(f"Error predicting for user {u}: {e}. Skipping user.")
            continue

    if not mat: # Check if mat is empty (e.g., all users were skipped)
         print("Warning: No predictions were generated.")
         return pd.DataFrame() 

    return pd.DataFrame(mat, index=users, columns=q_ids)

# ------------------------------------------------------------------
#  Difficulty from predictions (using 2PL)
# ------------------------------------------------------------------

def difficulty_from_predictions(pred_df):
    # Input pred_df has users as index, questions as columns
    if pred_df.empty:
        print("Prediction DataFrame is empty, cannot estimate difficulty.")
        return pd.DataFrame() # Return empty DataFrame
        
    long = (
        pred_df.reset_index() # user_id becomes a column
        .melt(id_vars="index", var_name="question_id", value_name="prob")
        .rename(columns={"index": "user_id"})
    )
    
    # Binarize the probabilities (as requested by user)
    long["pred_correct"] = (long["prob"] > 0.5).astype(int)
    
    # Use the new 2PL function
    # It expects user_col, question_col, correctness_col
    print("Estimating 2PL parameters from binarized predictions...")
    params_df = estimate_irt_2pl_params(
        response_df = long[["user_id","question_id","pred_correct"]],
        user_col = "user_id",
        question_col = "question_id",
        correctness_col = "pred_correct",
        # Add hyperparameters if needed, e.g., n_epochs=300, lr=0.01, reg_lambda=0.01
    )
    # The 2PL function returns difficulty and discrimination
    return params_df # Return the full DataFrame with difficulty and discrimination

# ------------------------------------------------------------------
#  Simple correlation comparison (Updated to handle new columns)
# ------------------------------------------------------------------

def compare_difficulty(orig_df, pred_df):
    # orig_df should have question_id and difficulty (let's assume it's from 2PL now)
    # pred_df has question_id, difficulty, discrimination
    merged = orig_df.merge(pred_df, on="question_id", how="inner", suffixes=('_orig', '_pred'))
    if merged.empty:
        print("Warning: No common questions found between original and predicted difficulties.")
        return {}
        
    # Compare the 'difficulty' columns
    pearson_corr = float(merged["difficulty_orig"].corr(merged["difficulty_pred"], method="pearson"))
    spearman_corr = float(merged["difficulty_orig"].corr(merged["difficulty_pred"], method="spearman"))
    
    # Optionally, report correlation with discrimination
    if 'discrimination_pred' in merged.columns:
         disc_corr_orig = float(merged["difficulty_orig"].corr(merged["discrimination_pred"], method="pearson"))
         disc_corr_pred = float(merged["difficulty_pred"].corr(merged["discrimination_pred"], method="pearson"))
         print(f"Correlation between original difficulty and predicted discrimination: {disc_corr_orig:.4f}")
         print(f"Correlation between predicted difficulty and predicted discrimination: {disc_corr_pred:.4f}")

    return dict(pearson=pearson_corr, spearman=spearman_corr, n=len(merged))

# ------------------------------------------------------------------
#  Persist any dict as json
# ------------------------------------------------------------------

def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

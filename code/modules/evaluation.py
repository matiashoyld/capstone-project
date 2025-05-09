"""Lightweight evaluation utilities."""

import os, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from modules.irt import estimate_irt_1pl_difficulty, estimate_irt_2pl_params

# ------------------------------------------------------------------
#  Build dataset for a given set of question IDs
# ------------------------------------------------------------------

def make_dataset(df, q_ids, preprocessors, combined_embeddings,
                 user_col, question_col, correctness_col):
    # --- Get necessary items from preprocessors ---    
    scaler = preprocessors["scaler"]
    uid2idx  = preprocessors["user_id_to_index"]
    original_numerical_cols = preprocessors['original_numerical_features'] # CORRECT KEY
    categorical_cols = preprocessors.get('categorical_features_encoded', [])
    ohe = preprocessors.get('ohe_encoder')
    ohe_feature_names = preprocessors.get('ohe_feature_names', [])
    final_numerical_cols = preprocessors['final_numerical_cols'] # Full list for scaler
    emb_dim  = preprocessors["embedding_dim"]

    sub = df[df[question_col].isin(q_ids)].copy()

    # --- Coerce original numerical columns and fill NaNs ---
    for col in original_numerical_cols:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')
    # NaNs from coercion will be handled by the fillna on data_numerical_combined or data_numerical_part

    # --- One-Hot Encode categorical features ---
    for col in categorical_cols:
        sub[col] = sub[col].fillna('Unknown').astype(str)

    if ohe and categorical_cols:
        ohe_features = ohe.transform(sub[categorical_cols])
        ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=sub.index)
        data_numerical_part = sub[original_numerical_cols].fillna(0) # Fill NaNs before concat
        data_numerical_combined = pd.concat([data_numerical_part, ohe_df], axis=1)
    else:
        data_numerical_combined = sub[original_numerical_cols].fillna(0) # Fill NaNs if no OHE

    # Ensure all final numerical columns are present and in correct order for the scaler
    for col in final_numerical_cols:
        if col not in data_numerical_combined.columns:
            data_numerical_combined[col] = 0 
    data_numerical_combined = data_numerical_combined[final_numerical_cols].fillna(0)

    # --- Scale numeric features ---
    X_num  = scaler.transform(data_numerical_combined)

    # --- User IDs ---
    X_user = sub[user_col].map(lambda u: uid2idx.get(u, 0)).values
    
    # --- Embeddings ---
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
    # num_cols should now refer to the final list of columns the scaler was trained on
    final_numerical_cols = preprocessors["final_numerical_cols"]
    original_numerical_cols = preprocessors['original_numerical_features']
    categorical_cols = preprocessors.get('categorical_features_encoded', [])
    ohe = preprocessors.get('ohe_encoder')
    ohe_feature_names = preprocessors.get('ohe_feature_names', [])
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
    # We need to apply the same OHE and coercion logic here as in make_dataset
    q_features_df = data_df[[question_col] + original_numerical_cols + categorical_cols].drop_duplicates(subset=[question_col]).set_index(question_col)
    q_features_ordered = q_features_df.reindex(q_ids)

    # Coerce original numerical columns
    for col in original_numerical_cols:
        q_features_ordered[col] = pd.to_numeric(q_features_ordered[col], errors='coerce')

    # OHE categorical columns
    for col in categorical_cols:
        q_features_ordered[col] = q_features_ordered[col].fillna('Unknown').astype(str)

    if ohe and categorical_cols:
        ohe_features_q = ohe.transform(q_features_ordered[categorical_cols])
        ohe_df_q = pd.DataFrame(ohe_features_q, columns=ohe_feature_names, index=q_features_ordered.index)
        q_numerical_part = q_features_ordered[original_numerical_cols].fillna(0)
        q_numerical_combined = pd.concat([q_numerical_part, ohe_df_q], axis=1)
    else:
        q_numerical_combined = q_features_ordered[original_numerical_cols].fillna(0)

    # Ensure all final columns for scaler, fill NaNs, and order
    for col in final_numerical_cols:
        if col not in q_numerical_combined.columns:
            q_numerical_combined[col] = 0
    q_numerical_combined = q_numerical_combined[final_numerical_cols].fillna(0)
    
    num_batch_scaled = scaler.transform(q_numerical_combined) # Shape: (len(q_ids), num_feature_count)

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

def difficulty_from_predictions(pred_df, irt_model_type="1pl"):
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
    print(f"Estimating {irt_model_type.upper()} parameters from binarized predictions...")
    if irt_model_type == "1pl":
        params_df = estimate_irt_1pl_difficulty(
            response_df = long[["user_id","question_id","pred_correct"]],
            user_col = "user_id",
            question_col = "question_id",
            correctness_col = "pred_correct",
            # Add hyperparameters if needed, e.g., n_epochs=300, lr=0.01, reg_lambda=0.01
        )
    elif irt_model_type == "2pl":
        params_df = estimate_irt_2pl_params(
            response_df = long[["user_id","question_id","pred_correct"]],
            user_col = "user_id",
            question_col = "question_id",
            correctness_col = "pred_correct",
            # Add hyperparameters if needed, e.g., n_epochs=300, lr=0.01, reg_lambda=0.01
        )
    else:
        raise ValueError(f"Unsupported irt_model_type: {irt_model_type}")
    return params_df # Return the full DataFrame with difficulty and discrimination

# ------------------------------------------------------------------
#  Simple correlation comparison (Updated to handle new columns)
# ------------------------------------------------------------------

def compare_difficulty(orig_df, pred_df, model_type="2pl"):
    # orig_df: question_id, difficulty_orig, (optional) discrimination_orig
    # pred_df: question_id, difficulty_pred, (optional) discrimination_pred
    
    # Define columns to merge based on model_type for original DataFrame
    orig_cols_to_merge = ["question_id", "difficulty"]
    if model_type == "2pl" and "discrimination" in orig_df.columns:
        orig_cols_to_merge.append("discrimination")
    
    # Define columns to merge for predicted DataFrame (pred_df should always have these if estimated)
    pred_cols_to_merge = ["question_id", "difficulty"]
    if "discrimination" in pred_df.columns: # Discrimination might not be there if pred_df is from 1PL
        pred_cols_to_merge.append("discrimination")

    # Select only necessary columns and rename for clarity before merge
    # This avoids issues if other columns with same names exist
    orig_df_renamed = orig_df[orig_cols_to_merge].rename(columns={
        "difficulty": "difficulty_orig",
        "discrimination": "discrimination_orig"
    })
    pred_df_renamed = pred_df[pred_cols_to_merge].rename(columns={
        "difficulty": "difficulty_pred",
        "discrimination": "discrimination_pred"
    })

    merged = orig_df_renamed.merge(pred_df_renamed, on="question_id", how="inner")
    
    if merged.empty:
        print("Warning: No common questions found between original and predicted difficulties after selecting columns.")
        return {}
        
    # Compare the 'difficulty' columns
    if 'difficulty_orig' not in merged.columns or 'difficulty_pred' not in merged.columns:
        print("Warning: Difficulty columns not found in merged DataFrame for comparison.")
        return {}
        
    pearson_corr = float(merged["difficulty_orig"].corr(merged["difficulty_pred"], method="pearson"))
    spearman_corr = float(merged["difficulty_orig"].corr(merged["difficulty_pred"], method="spearman"))
    
    results = dict(pearson=pearson_corr, spearman=spearman_corr, n=len(merged))

    # Optionally, report correlation with discrimination if comparing 2PL vs 2PL
    # And if both original and predicted discrimination columns are present after merge
    if model_type == "2pl" and 'discrimination_orig' in merged.columns and 'discrimination_pred' in merged.columns:
         disc_corr_orig_vs_pred_disc = float(merged["discrimination_orig"].corr(merged["discrimination_pred"], method="pearson"))
         results["disc_corr_orig_vs_pred_disc"] = disc_corr_orig_vs_pred_disc
         print(f"Correlation between original discrimination and predicted discrimination: {disc_corr_orig_vs_pred_disc:.4f}")

         # Correlation between original difficulty and predicted discrimination
         disc_corr_orig_diff_vs_pred_disc = float(merged["difficulty_orig"].corr(merged["discrimination_pred"], method="pearson"))
         results["disc_corr_orig_diff_vs_pred_disc"] = disc_corr_orig_diff_vs_pred_disc
         print(f"Correlation between original difficulty and predicted discrimination: {disc_corr_orig_diff_vs_pred_disc:.4f}")
         
         # Correlation between predicted difficulty and predicted discrimination
         disc_corr_pred_diff_vs_pred_disc = float(merged["difficulty_pred"].corr(merged["discrimination_pred"], method="pearson"))
         results["disc_corr_pred_diff_vs_pred_disc"] = disc_corr_pred_diff_vs_pred_disc
         print(f"Correlation between predicted difficulty and predicted discrimination: {disc_corr_pred_diff_vs_pred_disc:.4f}")

    return results

# ------------------------------------------------------------------
#  Persist any dict as json
# ------------------------------------------------------------------

def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------------------------------------------------
#  Setup paths
# ------------------------------------------------------------------

def setup_paths(run_dir, irt_model_type="1pl", nn_config_name="best_1pl_config_F_no_filter"):
    """Set up all file paths needed for the evaluation.

    Args:
        run_dir: Path to the directory containing run artifacts
        irt_model_type: '1pl' or '2pl' to specify which IRT model files to point to.
        nn_config_name: The specific configuration name for model and predicted params.
       
    Returns:
        dict: Dictionary of path strings
    """
    # Determine root directory more robustly
    if os.path.isabs(run_dir):
        parts = run_dir.split(os.sep)
        try:
            results_index = parts.index(next(p for p in reversed(parts) if "results" in p.lower()))
            root_dir = os.sep.join(parts[:results_index])
        except (StopIteration, ValueError):
            root_dir = os.path.abspath(os.path.join(os.path.dirname(run_dir), "..")) 
    else: 
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        code_dir = os.path.dirname(script_dir) 
        root_dir = os.path.dirname(code_dir) 
        run_dir = os.path.abspath(os.path.join(root_dir, run_dir.replace("../", ""))) 

    paths = {
        'data_dir': os.path.join(root_dir, "data", "zapien"),
        'run_dir': run_dir, 
        'plot_save_path': os.path.join(run_dir, f"efficiency_evaluation_rmse_{irt_model_type}_no_filter_run.png") 
    }
    
    paths['answers_file'] = os.path.join(paths['data_dir'], "answers.csv")
    
    # Specific naming for files from run_best_model_no_filter.py output
    is_no_filter_run = "_no_filter" in os.path.basename(run_dir) # Check the final directory name

    if is_no_filter_run:
        paths['embeddings_path'] = os.path.join(run_dir, "03_embeddings_no_filter.pkl")
        paths['holdout_ids_path'] = os.path.join(run_dir, "holdout_ids_no_filter.csv")
        paths['original_irt_path'] = os.path.join(run_dir, f"01_irt_{irt_model_type}_params_original_no_filter.csv")
        paths['qfeat_path'] = os.path.join(run_dir, "02_question_features_full_no_filter.csv")
        # nn_config_name is already specific for run_best_model_no_filter.py
        paths['model_weights_path'] = os.path.join(run_dir, "models", f"model_{nn_config_name}.keras") 
        paths['predicted_irt_file'] = os.path.join(run_dir, f"05_predicted_{irt_model_type}_params_{nn_config_name}.csv")
    else:
        # Fallback or logic for runs from optimization.py (might need further refinement if used)
        paths['embeddings_path'] = os.path.join(run_dir, "03_embeddings.pkl") 
        paths['holdout_ids_path'] = os.path.join(run_dir, "holdout_ids.csv") 
        paths['original_irt_path'] = os.path.join(run_dir, f"01_irt_{irt_model_type}_params.csv")
        paths['qfeat_path'] = os.path.join(run_dir, "02_question_features_full.csv")
        # This part for optimization.py output would need to handle the config name in predicted files
        # e.g., f"05_predicted_{irt_model_type}_params_{nn_config_name}_raw_probs_False.csv"
        paths['model_weights_path'] = os.path.join(run_dir, "models", f"model_config_{nn_config_name.replace('_binIRT','')}.keras")
        paths['predicted_irt_file'] = os.path.join(run_dir, f"05_predicted_{irt_model_type}_params_{nn_config_name}.csv") # Placeholder, needs specific suffix from opto script

    # Check existence of critical files
    for key, path_val in paths.items():
        if "path" in key and not os.path.exists(path_val) and key not in ['plot_save_path']: # plot_save_path is an output
            print(f"Warning: Path for '{key}' not found: {path_val}")
            if key == 'original_irt_path' or key == 'predicted_irt_file':
                print(f"Warning: Critical file {key} missing. RMSE evaluation might fail.")

    return paths

# --- ADDED: calculate_rmse function ---
def calculate_rmse(true_difficulty_df, estimated_difficulty_df, question_col, irt_col):
    """
    Calculate RMSE between true and estimated difficulty parameters after aligning scales.
    
    Args:
        true_difficulty_df: DataFrame with ground truth difficulties
        estimated_difficulty_df: DataFrame with estimated difficulties
        question_col: Name of the column containing question IDs
        irt_col: Name of the column containing difficulty parameters
        
    Returns:
        float: RMSE value, or np.nan if calculation is not possible
    """
    true_difficulty_df = true_difficulty_df.copy()
    estimated_difficulty_df = estimated_difficulty_df.copy()
    true_difficulty_df[question_col] = true_difficulty_df[question_col].astype(str)
    estimated_difficulty_df[question_col] = estimated_difficulty_df[question_col].astype(str)

    merged = pd.merge(
        true_difficulty_df[[question_col, irt_col]],
        estimated_difficulty_df[[question_col, irt_col]],
        on=question_col,
        suffixes=("_true", "_est")
    )
    
    if merged.empty or len(merged) < 2:
        try: logger.warning("RMSE: Not enough matching questions (<2) for calculation.")
        except NameError: print("Warning: RMSE: Not enough matching questions (<2) for calculation.")
        return np.nan

    true_col_name = f"{irt_col}_true"
    est_col_name = f"{irt_col}_est"
    
    if true_col_name not in merged.columns or est_col_name not in merged.columns:
        try: logger.error(f"RMSE: Required columns not found after merge: {true_col_name}, {est_col_name}")
        except NameError: print(f"Error: RMSE: Required columns not found after merge: {true_col_name}, {est_col_name}")
        return np.nan

    # Drop rows where either true or estimated difficulty is NaN before any calculations
    merged_clean = merged[[true_col_name, est_col_name]].dropna()
    if len(merged_clean) < 2:
        try: logger.warning("RMSE: Not enough non-NaN pairs (<2) after dropna.")
        except NameError: print("Warning: RMSE: Not enough non-NaN pairs (<2) after dropna.")
        return np.nan

    common_true_difficulties = merged_clean[true_col_name]
    common_est_difficulties = merged_clean[est_col_name]

    # Check for constant series after cleaning NaN values
    if common_true_difficulties.nunique() <= 1 and common_est_difficulties.nunique() <= 1:
        if common_true_difficulties.iloc[0] == common_est_difficulties.iloc[0]:
            return 0.0 # Both constant and equal
        else:
            # Both constant but different, RMSE will reflect this difference without scaling
            pass 
    elif common_true_difficulties.nunique() <= 1 or common_est_difficulties.nunique() <= 1:
        # One is constant, the other is not. RMSE will be high, scaling might not be appropriate or helpful.
        try: logger.warning("RMSE: One difficulty series is constant, the other is not. RMSE may be misleading.")
        except NameError: print("Warning: RMSE: One difficulty series is constant, the other is not. RMSE may be misleading.")
        # Proceed without scaling in this specific edge case if one is constant and other is not.
        rmse = np.sqrt(mean_squared_error(common_true_difficulties, common_est_difficulties))
        return rmse

    # --- Align scales for RMSE calculation if both are variable ---
    # Scale estimated to match true for RMSE calculation
    mean_true = common_true_difficulties.mean()
    std_true = common_true_difficulties.std()
    mean_est = common_est_difficulties.mean()
    std_est = common_est_difficulties.std()

    if std_true < 1e-9 or std_est < 1e-9: # Check for effectively constant series (very small std dev)
        # If one or both are effectively constant after all, calculate direct RMSE or handle as constant case
        if std_true < 1e-9 and std_est < 1e-9 and abs(mean_true - mean_est) < 1e-9:
            return 0.0
        # If one is constant and the other isn't, or both constant but different means
        # it might be better to return the unscaled RMSE or NaN if comparison is not meaningful.
        try: logger.warning("RMSE: One or both difficulty series have near-zero std dev. Calculating direct RMSE.")
        except NameError: print("Warning: RMSE: One or both difficulty series have near-zero std dev. Calculating direct RMSE.")
        rmse = np.sqrt(mean_squared_error(common_true_difficulties, common_est_difficulties))
        return rmse
            
    est_scaled_for_rmse = (common_est_difficulties - mean_est) / std_est * std_true + mean_true
    rmse = np.sqrt(mean_squared_error(common_true_difficulties, est_scaled_for_rmse))
    
    return rmse

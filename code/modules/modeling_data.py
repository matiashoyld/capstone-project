"""Tiny helpers for splitting data and preparing NN inputs.

Heavy lifting (e.g., orchestration, parameter choice) is done in the
notebook; these functions focus on **single, reusable actions**.
"""

import os, pickle
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------------------------------
#  Split questions into train / val / test
# --------------------------------------------------

def stratified_question_split_3way(
    df,
    question_col,
    stratify_cols,
    test_size = 0.1,
    val_size  = 0.2,
    random_state = 42,
    n_bins = 5,
):
    """Return (train_ids, val_ids, test_ids).

    * We aggregate correctness / difficulty at the question level.
    * Numeric columns are binned into quantiles, others are used as‑is.
    * A second split carves validation out of the train+val pool.
    """
    agg = {
        col: ("mean" if pd.api.types.is_numeric_dtype(df[col]) else "first")
        for col in stratify_cols
    }
    qmetrics = df.groupby(question_col).agg(agg).reset_index()

    codes = []
    for col in stratify_cols:
        if pd.api.types.is_numeric_dtype(qmetrics[col]):
            # bin numeric column
            uniq = qmetrics[col].nunique()
            if uniq > 1:
                bins = min(n_bins, uniq)
                binned = pd.qcut(qmetrics[col], bins, labels=False, duplicates="drop")
                codes.append(binned.astype(str))
            else:
                codes.append(pd.Series("0", index=qmetrics.index))
        else:
            codes.append(qmetrics[col].astype(str))

    qmetrics["strat"] = pd.concat(codes, axis=1).agg("_".join, axis=1)

    train_val, test = train_test_split(
        qmetrics,
        test_size = test_size,
        random_state = random_state,
        stratify = qmetrics["strat"],
    )
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size = val_fraction,
        random_state = random_state,
        stratify = train_val["strat"],
    )

    return (
        train[question_col].tolist(),
        val[question_col].tolist(),
        test[question_col].tolist(),
    )

# --------------------------------------------------
#  Prepare NN inputs
# --------------------------------------------------

def prepare_nn_datasets(
    merged_df,
    combined_embeddings,
    train_q_ids,
    val_q_ids,
    user_col,
    question_col,
    correctness_col,
    numerical_feature_cols,
    categorical_feature_cols_to_encode,
    embedding_dim,
):
    """Build numpy arrays for Keras.

    Returns (train_ds, train_y, val_ds, val_y, preprocessors, final_numerical_cols)
    """
    train_df = merged_df[merged_df[question_col].isin(train_q_ids)].copy()
    val_df   = merged_df[merged_df[question_col].isin(val_q_ids)].copy()

    # --- NEW: Coerce original numerical columns to numeric, handling '#ERROR!' ---
    for col in numerical_feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce')
    # NaNs resulting from coercion (e.g., from '#ERROR!') will be handled by fillna later.

    # --- NEW: Handle potential NaNs in categorical columns before OHE ---
    for col in categorical_feature_cols_to_encode:
        train_df[col] = train_df[col].fillna('Unknown').astype(str)
        val_df[col] = val_df[col].fillna('Unknown').astype(str)
    
    # --- NEW: One-Hot Encode specified categorical features ---
    ohe_transformers = []
    if categorical_feature_cols_to_encode:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Fit OHE on training data only
        ohe.fit(train_df[categorical_feature_cols_to_encode])
        
        train_ohe_features = ohe.transform(train_df[categorical_feature_cols_to_encode])
        val_ohe_features = ohe.transform(val_df[categorical_feature_cols_to_encode])
        
        ohe_feature_names = ohe.get_feature_names_out(categorical_feature_cols_to_encode)
        
        train_ohe_df = pd.DataFrame(train_ohe_features, columns=ohe_feature_names, index=train_df.index)
        val_ohe_df = pd.DataFrame(val_ohe_features, columns=ohe_feature_names, index=val_df.index)
        
        # Combine OHE features with original numerical features
        # Fill NaNs in numerical_feature_cols (which might include NaNs from coerced errors) before concatenating
        train_numerical_part = train_df[numerical_feature_cols].fillna(0)
        val_numerical_part = val_df[numerical_feature_cols].fillna(0)
        
        train_df_numerical_combined = pd.concat([train_numerical_part, train_ohe_df], axis=1)
        val_df_numerical_combined = pd.concat([val_numerical_part, val_ohe_df], axis=1)
        
        final_numerical_cols = numerical_feature_cols + list(ohe_feature_names)
    else:
        ohe = None
        ohe_feature_names = []
        # Fill NaNs in numerical_feature_cols (which might include NaNs from coerced errors)
        train_df_numerical_combined = train_df[numerical_feature_cols].fillna(0)
        val_df_numerical_combined = val_df[numerical_feature_cols].fillna(0)
        final_numerical_cols = numerical_feature_cols

    # ------------------------------------------------ numeric features scaling
    scaler = StandardScaler()
    if final_numerical_cols: # Check if there are any numerical features to scale
        train_num = scaler.fit_transform(train_df_numerical_combined[final_numerical_cols])
        val_num   = scaler.transform(val_df_numerical_combined[final_numerical_cols])
    else: # No numerical features, create empty arrays with correct shape for Keras
        train_num = np.empty((len(train_df), 0))
        val_num = np.empty((len(val_df), 0))
        scaler = None # No scaler was fit or used

    # ------------------------------------------------ user ids
    users = pd.concat([train_df[user_col], val_df[user_col]]).unique()
    uid2idx = {u: i+1 for i, u in enumerate(users)}  # reserve 0 for OOV
    train_user = train_df[user_col].map(uid2idx).fillna(0).astype(int).values
    val_user   = val_df[user_col].map(uid2idx).fillna(0).astype(int).values

    # ------------------------------------------------ embeddings
    emb_map = combined_embeddings.get("formatted_embeddings", {})
    default_emb = np.zeros(embedding_dim)
    train_emb = np.vstack([emb_map.get(q, default_emb) for q in train_df[question_col]])
    val_emb   = np.vstack([emb_map.get(q, default_emb) for q in val_df[question_col]])

    # ------------------------------------------------ labels
    train_y = train_df[correctness_col].astype(int).values
    val_y   = val_df[correctness_col].astype(int).values

    # ------------------------------------------------ assemble
    train_ds = {
        "user_input": train_user,
        "numerical_input": train_num,
        "embedding_input": train_emb,
    }
    val_ds = {
        "user_input": val_user,
        "numerical_input": val_num,
        "embedding_input": val_emb,
    }

    preprocessors = dict(
        scaler = scaler,
        user_id_to_index = uid2idx,
        user_vocab_size = len(uid2idx) + 1,
        embedding_dim = embedding_dim,
        original_numerical_features = numerical_feature_cols,
        categorical_features_encoded = categorical_feature_cols_to_encode,
        ohe_encoder = ohe,
        ohe_feature_names = list(ohe_feature_names),
        final_numerical_cols = final_numerical_cols
    )

    return train_ds, train_y, val_ds, val_y, preprocessors, final_numerical_cols

# --------------------------------------------------
#  Dump preprocessors
# --------------------------------------------------

def save_preprocessors(preprocessors, results_dir, filename = "nn_preprocessors.pkl"):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), "wb") as f:
        pickle.dump(preprocessors, f)

def load_preprocessors(load_dir: str) -> dict:
    """Loads fitted preprocessors from disk."""
    path = os.path.join(load_dir, "nn_preprocessors.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessors file not found at {path}")
    with open(path, "rb") as f:
        preprocessors = pickle.load(f)
    print(f"✅ Loaded preprocessors from {path}")
    return preprocessors

# --- NEW: make_dataset function for test/evaluation data ---
def make_dataset(
    df, 
    q_ids, 
    preprocessors, 
    combined_embeddings, 
    user_col, 
    question_col, 
    correctness_col
):
    """Prepares a dataset (e.g., test set) using fitted preprocessors."""
    data_df = df[df[question_col].isin(q_ids)].copy()

    original_numerical_cols = preprocessors['original_numerical_features']
    categorical_cols = preprocessors.get('categorical_features_encoded', [])
    ohe = preprocessors.get('ohe_encoder')
    ohe_feature_names = preprocessors.get('ohe_feature_names', [])
    final_numerical_cols = preprocessors['final_numerical_cols'] # Get the full list of columns for the scaler

    # Coerce original numerical columns to numeric, handling '#ERROR!'
    for col in original_numerical_cols:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    # Fill NaNs for categorical columns before transforming
    for col in categorical_cols:
        data_df[col] = data_df[col].fillna('Unknown').astype(str)

    if ohe and categorical_cols:
        ohe_features = ohe.transform(data_df[categorical_cols])
        ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=data_df.index)
        
        data_numerical_part = data_df[original_numerical_cols].fillna(0)
        data_numerical_combined = pd.concat([data_numerical_part, ohe_df], axis=1)
    else:
        data_numerical_combined = data_df[original_numerical_cols].fillna(0)
    
    # Ensure all final numerical columns are present and in correct order for the scaler, fill NaNs
    for col in final_numerical_cols:
        if col not in data_numerical_combined.columns:
            data_numerical_combined[col] = 0 
    # Ensure the columns are in the same order as when the scaler was fitted.
    # If final_numerical_cols is empty, this will correctly result in an empty DataFrame slice.
    data_numerical_combined = data_numerical_combined.reindex(columns=final_numerical_cols, fill_value=0)

    # --- Numeric features: Scale using fitted scaler --- 
    scaler = preprocessors['scaler']
    if scaler and final_numerical_cols: # Check if scaler exists and there are columns to transform
        num_features = scaler.transform(data_numerical_combined) 
    else: # No scaler used or no numerical features
        num_features = np.empty((len(data_df), 0))

    # --- User IDs: Map using fitted mapping ---
    uid2idx = preprocessors['user_id_to_index']
    user_features = data_df[user_col].map(uid2idx).fillna(0).astype(int).values

    # --- Embeddings ---
    emb_map = combined_embeddings.get("formatted_embeddings", {})
    default_emb = np.zeros(preprocessors['embedding_dim'])
    emb_features = np.vstack([emb_map.get(q, default_emb) for q in data_df[question_col]])

    # --- Labels ---
    labels = data_df[correctness_col].astype(int).values

    dataset_inputs = {
        "user_input": user_features,
        "numerical_input": num_features,
        "embedding_input": emb_features,
    }
    
    return dataset_inputs, labels

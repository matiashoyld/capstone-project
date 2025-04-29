"""Tiny helpers for splitting data and preparing NN inputs.

Heavy lifting (e.g., orchestration, parameter choice) is done in the
notebook; these functions focus on **single, reusable actions**.
"""

import os, pickle
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    embedding_dim,
):
    """Build numpy arrays for Keras.

    Returns (train_ds, train_y, val_ds, val_y, preprocessors)
    """
    train_df = merged_df[merged_df[question_col].isin(train_q_ids)].copy()
    val_df   = merged_df[merged_df[question_col].isin(val_q_ids)].copy()

    # ------------------------------------------------ numeric features
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[numerical_feature_cols])
    val_num   = scaler.transform(val_df[numerical_feature_cols])

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
        numerical_features = numerical_feature_cols,
    )

    return train_ds, train_y, val_ds, val_y, preprocessors

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

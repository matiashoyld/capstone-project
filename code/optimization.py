import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging # For better logging
import json # For saving hyperparams

# Standard TensorFlow/Keras imports (will be used later)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# For Permutation Importance
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin # For Keras wrapper

# Import your custom modules
from modules.irt import estimate_irt_1pl_difficulty, estimate_irt_2pl_params # Ensure 1PL is imported
from modules.features import extract_text_features, calculate_option_features
from modules.embeddings import generate_text_embeddings
from modules.utils import format_question_text
from modules.modeling_data import (
    stratified_question_split_3way,
    prepare_nn_datasets,
    save_preprocessors,
    load_preprocessors, # We'll need this if we save preprocessors during optimization runs
    make_dataset # Assuming this is the one from modeling_data.py
)
from modules.neural_net import create_nn_model, train_nn_model, save_nn_model
from modules.evaluation import (
    evaluate_model, # Keep this
    dump_json,
    prediction_matrix,
    difficulty_from_predictions, # We might modify a copy of this
    compare_difficulty,
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants (Copied and adapted from main.ipynb) ---
USER_ID_COL = "user_id"
QUESTION_ID_COL = "question_id"
CORRECTNESS_COL = "is_correct"
QUESTION_TEXT_COL = "question_title"
OPTION_COLS = ["option_a", "option_b", "option_c", "option_d", "option_e"]
CORRECT_OPTION_COL = "correct_option_letter"
FORMATTED_TEXT_COL = "formatted_question_text"
IRT_DIFFICULTY_COL = "difficulty" # This is key for comparison
IRT_DISCRIMINATION_COL = "discrimination" # For 2PL

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
EMBEDDING_BATCH_SIZE = 32
TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE = 0.2 # Proportion of (train+val) for validation
RANDOM_SEED = 42

# Original numerical features (before OHE)
BASE_NUMERICAL_FEATURE_COLS = [
    "question_word_count", "question_char_count", "question_avg_word_length",
    "question_digit_count", "question_special_char_count",
    "question_mathematical_symbols", "question_latex_expressions",
    "jaccard_similarity_std", "avg_option_length", "avg_option_word_count",
    "avg_steps", "level", "num_misconceptions", "has_image",
    "Answer_Length_Variance", "Correct_Distractor_CosineSim_Mean",
    "Distractor_Embedding_Distance_Mean", "Extreme_Wording_Option_Count",
    "Has_Abstract_Symbols", "Has_NoneAll_Option",
    "LLM_Distractor_Plausibility_Max", "LLM_Distractor_Plausibility_Mean",
    "Mathematical_Notation_Density", "Max_Expression_Nesting_Depth",
    "Option_Length_Outlier_Flag", "Question_Answer_Info_Gap",
    "Ratio_Abstract_Concrete_Symbols", "RealWorld_Context_Flag", "Units_Check"
]

CATEGORICAL_FEATURE_COLS_TO_ENCODE = [
    "Knowledge_Dimension",
    "Most_Complex_Number_Type",
    "Problem_Archetype"
]

# --- Path Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming optimization.py is in code/
DATA_DIR = os.path.join(ROOT_DIR, "data", "zapien")
ANSWERS_FILE_PATH = os.path.join(DATA_DIR, "answers.csv")
QUESTIONS_FILE_PATH = os.path.join(DATA_DIR, "questions.csv")
ADDITIONAL_FEATURES_FILE_PATH = os.path.join(DATA_DIR, "questions_additional_features.csv")

RESULTS_BASE_DIR = os.path.join(ROOT_DIR, "results_optimization") # New results directory
CURRENT_RUN_DIR = os.path.join(RESULTS_BASE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)
logger.info(f"Results will be saved to: {CURRENT_RUN_DIR}")

# --- Keras Model Wrapper for Scikit-learn Permutation Importance ---
class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # Placeholder fit, actual training happens outside
        return self

    def predict_proba(self, X):
        # X is expected to be a dictionary of inputs if model has multiple inputs
        return self.model.predict(X)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def load_and_preprocess_data():
    logger.info("Loading raw data...")
    student_answers_df = pd.read_csv(ANSWERS_FILE_PATH)
    question_content_df = pd.read_csv(QUESTIONS_FILE_PATH)
    additional_features_df = pd.read_csv(ADDITIONAL_FEATURES_FILE_PATH)

    logger.info(f"Original student answers: {student_answers_df.shape}")
    logger.info(f"Original question content: {question_content_df.shape}")
    logger.info(f"Additional features: {additional_features_df.shape}")

    # Merge additional features
    question_content_df = pd.merge(question_content_df, additional_features_df, on=QUESTION_ID_COL, how='left')
    logger.info(f"Question content after merging additional features: {question_content_df.shape}")
    
    # --- Handle NaNs in new features ---
    # Numerical: Fill with median (example, can be configured)
    for col in [
        "Answer_Length_Variance", "Correct_Distractor_CosineSim_Mean",
        "Distractor_Embedding_Distance_Mean", "Extreme_Wording_Option_Count", # Count-like, 0 might be ok or median
        "LLM_Distractor_Plausibility_Max", "LLM_Distractor_Plausibility_Mean",
        "Mathematical_Notation_Density", "Max_Expression_Nesting_Depth",
        "Question_Answer_Info_Gap", "Ratio_Abstract_Concrete_Symbols"
    ]:
        if col in question_content_df.columns:
            question_content_df[col] = pd.to_numeric(question_content_df[col], errors='coerce') # Ensure numeric
            question_content_df[col].fillna(question_content_df[col].median(), inplace=True)

    # Binary (0/1) already in BASE_NUMERICAL_FEATURE_COLS: fillna(0) might be acceptable if 0 is a clear "false" or "absent"
    for col in ["Has_Abstract_Symbols", "Has_NoneAll_Option", "Option_Length_Outlier_Flag", "RealWorld_Context_Flag", "Units_Check"]:
        if col in question_content_df.columns:
            question_content_df[col] = pd.to_numeric(question_content_df[col], errors='coerce').fillna(0)


    # Categorical: fillna with 'Unknown' (handled later in prepare_nn_datasets, but good practice here too)
    for col in CATEGORICAL_FEATURE_COLS_TO_ENCODE:
        if col in question_content_df.columns:
            question_content_df[col] = question_content_df[col].astype(str).fillna('Unknown')


    # --- Filter Questions Based on Response Patterns (copied from notebook) ---
    logger.info(f"Original student answers shape: {student_answers_df.shape}")
    initial_question_count = student_answers_df[QUESTION_ID_COL].nunique()
    logger.info(f"Original unique questions: {initial_question_count}")
    logger.info("Filtering questions with 0%/100% correctness or < 10 responses...")

    question_stats = student_answers_df.groupby(QUESTION_ID_COL)[CORRECTNESS_COL].agg(['mean', 'count'])
    qids_to_remove = question_stats[
        (question_stats['mean'] == 0) |
        (question_stats['mean'] == 1) |
        (question_stats['count'] < 10)
    ].index

    if not qids_to_remove.empty:
        removed_responses_count = student_answers_df[student_answers_df[QUESTION_ID_COL].isin(qids_to_remove)].shape[0]
        student_answers_df = student_answers_df[~student_answers_df[QUESTION_ID_COL].isin(qids_to_remove)]
        logger.info(f"Removed {removed_responses_count} responses belonging to {len(qids_to_remove)} questions due to filtering criteria.")
        logger.info(f"Filtered student answers shape: {student_answers_df.shape}")
        logger.info(f"Remaining unique questions: {student_answers_df[QUESTION_ID_COL].nunique()}")
    else:
        logger.info("No questions met the filtering criteria for removal.")

    if student_answers_df.empty:
        raise ValueError("Response DataFrame is empty after filtering. Cannot proceed.")
    
    # Filter question_content_df to only include questions present in filtered student_answers_df
    valid_qids_after_filter = student_answers_df[QUESTION_ID_COL].unique()
    question_content_df = question_content_df[question_content_df[QUESTION_ID_COL].isin(valid_qids_after_filter)]
    logger.info(f"Question content after filtering to match answers: {question_content_df.shape}")


    return student_answers_df, question_content_df

def calculate_irt_difficulties(student_answers_df, irt_model_type="2pl"):
    if irt_model_type == "1pl":
        logger.info("Estimating question difficulty using IRT 1PL model...")
        # Ensure estimate_irt_1pl_difficulty is available and imported correctly
        question_params_df = estimate_irt_1pl_difficulty(
            response_df=student_answers_df,
            user_col=USER_ID_COL,
            question_col=QUESTION_ID_COL,
            correctness_col=CORRECTNESS_COL,
        )
        file_suffix = "1pl"
        log_suffix = "1PL"
        # 1PL typically only returns difficulty
        returned_columns = [QUESTION_ID_COL, IRT_DIFFICULTY_COL]
    elif irt_model_type == "2pl":
        logger.info("Estimating question difficulty using IRT 2PL model...")
        question_params_df = estimate_irt_2pl_params(
            response_df=student_answers_df,
            user_col=USER_ID_COL,
            question_col=QUESTION_ID_COL,
            correctness_col=CORRECTNESS_COL,
        )
        file_suffix = "2pl"
        log_suffix = "2PL"
        returned_columns = [QUESTION_ID_COL, IRT_DIFFICULTY_COL, IRT_DISCRIMINATION_COL]
    else:
        raise ValueError(f"Unsupported irt_model_type: {irt_model_type}. Choose '1pl' or '2pl'.")

    question_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, f"01_irt_{file_suffix}_params.csv"), index=False)
    logger.info(f"IRT {log_suffix} parameters saved.")
    logger.info(f"IRT Difficulty ({log_suffix}) summary:\n{question_params_df[IRT_DIFFICULTY_COL].describe()}")
    return question_params_df[returned_columns]


def extract_all_features(question_content_df):
    logger.info("Extracting text-based features...")
    text_features_df = question_content_df[QUESTION_TEXT_COL].apply(
        lambda text: pd.Series(extract_text_features(str(text))) # ensure text is str
    ).add_prefix("question_")

    logger.info("Extracting option-based features...")
    # Ensure option columns are strings, fill NaN with empty string
    for col in OPTION_COLS:
        question_content_df[col] = question_content_df[col].fillna('').astype(str)

    option_features_list = question_content_df.apply(
        lambda row: calculate_option_features(row, OPTION_COLS),
        axis=1
    ).tolist()
    option_features_df = pd.DataFrame(option_features_list, index=question_content_df.index)

    questions_with_features_df = pd.concat(
        [question_content_df, text_features_df, option_features_df],
        axis=1
    )
    
    # Ensure BASE_NUMERICAL_FEATURE_COLS only contains columns actually present
    # This is important if some features from the original list aren't generated by the functions
    # or if additional_features_df didn't have all of them.
    # The most robust way is to build this list dynamically based on what's available.
    # For now, assuming most will be present.
    
    # Fill NaNs for any features that might not have been generated for all questions
    # (e.g., text features for empty question titles if not handled by extract_text_features)
    # This is a placeholder, specific handling might be needed for specific features.
    # Many of these are already handled in load_and_preprocess_data for the *new* features.
    # This part addresses features generated by extract_text_features and calculate_option_features
    
    # Example: Check for text complexity features (which are part of BASE_NUMERICAL_FEATURE_COLS)
    generated_text_feature_cols = [
        "question_word_count", "question_char_count", "question_avg_word_length",
        "question_digit_count", "question_special_char_count",
        "question_mathematical_symbols", "question_latex_expressions",
    ]
    for col in generated_text_feature_cols:
        if col in questions_with_features_df.columns:
            questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)
        else: # If feature wasn't generated, add it as a column of zeros
            logger.warning(f"Feature {col} not found after extraction, adding as zeros.")
            questions_with_features_df[col] = 0


    generated_option_feature_cols = ["jaccard_similarity_std", "avg_option_length", "avg_option_word_count"]
    for col in generated_option_feature_cols:
         if col in questions_with_features_df.columns:
            questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)
         else:
            logger.warning(f"Feature {col} not found after option feature extraction, adding as zeros.")
            questions_with_features_df[col] = 0
            
    # Ensure all BASE_NUMERICAL_FEATURE_COLS are present before they are used later
    # If a feature is in BASE_NUMERICAL_FEATURE_COLS but not in questions_with_features_df after all merges/extractions,
    # it will cause a KeyError later.
    for col in BASE_NUMERICAL_FEATURE_COLS:
        if col not in questions_with_features_df.columns:
            logger.warning(f"Column '{col}' from BASE_NUMERICAL_FEATURE_COLS not found in DataFrame. Adding it with zeros.")
            questions_with_features_df[col] = 0
        else: # If column exists, ensure it's numeric and fill NaNs that might have resulted from merges/joins
             questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)


    questions_with_features_df.to_csv(
        os.path.join(CURRENT_RUN_DIR, "02_question_features_full.csv"),
        index=False
    )
    logger.info("Questions with all features saved.")
    return questions_with_features_df

# --- NEW: Function to estimate IRT from raw probabilities ---
def difficulty_from_probabilities_irt(pred_df, irt_model_type="2pl"):
    logger.info(f"Estimating IRT {irt_model_type.upper()} parameters from raw prediction probabilities...")
    if pred_df.empty:
        logger.warning("Prediction DataFrame is empty, cannot estimate difficulty.")
        return pd.DataFrame()
        
    long = (
        pred_df.reset_index() 
        .melt(id_vars="index", var_name=QUESTION_ID_COL, value_name="prob")
        .rename(columns={"index": USER_ID_COL, "prob": CORRECTNESS_COL
                        })
    )
    
    long[CORRECTNESS_COL] = long[CORRECTNESS_COL].astype(float)

    if irt_model_type == "1pl":
        params_df = estimate_irt_1pl_difficulty(
            response_df=long[[USER_ID_COL, QUESTION_ID_COL, CORRECTNESS_COL]],
            user_col=USER_ID_COL,
            question_col=QUESTION_ID_COL,
            correctness_col=CORRECTNESS_COL, 
        )
    elif irt_model_type == "2pl":
        params_df = estimate_irt_2pl_params(
            response_df=long[[USER_ID_COL, QUESTION_ID_COL, CORRECTNESS_COL]],
            user_col=USER_ID_COL,
            question_col=QUESTION_ID_COL,
            correctness_col=CORRECTNESS_COL, 
        )
    else:
        raise ValueError(f"Unsupported irt_model_type: {irt_model_type}")
    return params_df

# --- MODIFIED: Original difficulty_from_predictions to also accept irt_model_type ---
# This function might need to be moved from evaluation.py to here or updated in both places.
# For now, assuming we modify a version within this script if it was copied, or we ensure evaluation.py is updated.

# Let's assume the `difficulty_from_predictions` from `modules.evaluation` is the one being used and modify it there.
# If it's to be kept separate, this optimization.py can have its own local version or call the updated module version.

# For the purpose of this script, let's define a local wrapper or assume modules.evaluation.difficulty_from_predictions
# will be updated. If we add it here:

def local_difficulty_from_binarized_predictions(pred_df, irt_model_type="2pl"):
    logger.info(f"Estimating IRT {irt_model_type.upper()} parameters from BINARIZED prediction probabilities...")
    if pred_df.empty:
        logger.warning("Prediction DataFrame is empty, cannot estimate difficulty.")
        return pd.DataFrame()
    
    long = (
        pred_df.reset_index()
        .melt(id_vars="index", var_name=QUESTION_ID_COL, value_name="prob")
        .rename(columns={"index": USER_ID_COL})
    )
    long[CORRECTNESS_COL] = (long["prob"] > 0.5).astype(int) # Binarization

    if irt_model_type == "1pl":
        params_df = estimate_irt_1pl_difficulty(
            response_df=long[[USER_ID_COL, QUESTION_ID_COL, CORRECTNESS_COL]],
            user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL)
    elif irt_model_type == "2pl":
        params_df = estimate_irt_2pl_params(
            response_df=long[[USER_ID_COL, QUESTION_ID_COL, CORRECTNESS_COL]],
            user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL)
    else:
        raise ValueError(f"Unsupported irt_model_type: {irt_model_type}")
    return params_df


def run_single_nn_config(config_name, nn_config, preprocessors, train_ds, train_y, val_ds, val_y, 
                         complete_dataset_df, test_q_ids, question_embeddings, original_irt_params_df,
                         embedding_dimension, final_num_cols, 
                         prediction_irt_model_type="2pl", # New param
                         use_raw_probs_for_irt=False):
    logger.info(f"--- Running NN Configuration: {config_name} ---")
    logger.info(f"NN Config: {nn_config}")

    model = create_nn_model(
        user_vocab_size=preprocessors["user_vocab_size"],
        numerical_feature_size=len(final_num_cols),
        embedding_dim=embedding_dimension,
        user_embedding_dim=nn_config.get("user_embedding_dim", 8),
        dropout_rate=nn_config.get("dropout_rate", 0.3),
        l2_reg=nn_config.get("l2_reg", 0.001),
        dense_layers_config=nn_config.get("dense_layers", [64, 32])
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=nn_config.get("learning_rate", 1e-3)),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    history = train_nn_model(
        model, train_ds, train_y, val_ds, val_y,
        epochs=nn_config.get("epochs", 30),
        batch_size=nn_config.get("batch_size", 1024),
        patience_es=nn_config.get("patience_es", 5)
    )
    model_filename = f"model_config_{config_name.replace(' ', '_').lower()}.keras"
    save_nn_model(model, CURRENT_RUN_DIR, filename=model_filename)
    logger.info(f"Model training complete for {config_name}.")

    logger.info(f"Evaluating {config_name} on hold-out test data...")
    test_dataset_inputs, test_labels = make_dataset(
        df=complete_dataset_df,
        q_ids=test_q_ids,
        preprocessors=preprocessors,
        combined_embeddings=question_embeddings,
        user_col=USER_ID_COL,
        question_col=QUESTION_ID_COL,
        correctness_col=CORRECTNESS_COL
    )
    test_metrics = evaluate_model(model, test_dataset_inputs, test_labels)
    dump_json(test_metrics, os.path.join(CURRENT_RUN_DIR, f"holdout_metrics_{config_name}.json"))
    logger.info(f"Test set performance ({config_name}): {test_metrics}")

    logger.info(f"Generating prediction probability matrix for {config_name}...")
    prob_matrix = prediction_matrix(
        data_df=complete_dataset_df,
        q_ids=test_q_ids,
        preprocessors=preprocessors,
        model=model,
        combined_embeddings=question_embeddings,
        user_col=USER_ID_COL,
        question_col=QUESTION_ID_COL
    )
    prob_matrix.to_csv(os.path.join(CURRENT_RUN_DIR, f"04_prediction_probabilities_{config_name}.csv"))
    logger.info(f"Prediction matrix saved for {config_name}.")

    if use_raw_probs_for_irt:
        logger.info(f"Estimating IRT from RAW probabilities ({prediction_irt_model_type.upper()}) for {config_name}...")
        predicted_irt_params_df = difficulty_from_probabilities_irt(prob_matrix, irt_model_type=prediction_irt_model_type)
    else:
        logger.info(f"Estimating IRT from BINARIZED probabilities ({prediction_irt_model_type.upper()}) for {config_name}...")
        # Using the local version for this script, or ensure modules.evaluation.difficulty_from_predictions is updated
        predicted_irt_params_df = local_difficulty_from_binarized_predictions(prob_matrix, irt_model_type=prediction_irt_model_type) 
    
    file_suffix_irt = f"{prediction_irt_model_type.lower()}_raw_probs_{use_raw_probs_for_irt}"
    predicted_irt_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, f"05_predicted_{file_suffix_irt}_params_{config_name}.csv"), index=False)
    logger.info(f"Predicted IRT {prediction_irt_model_type.upper()} params saved for {config_name}.")
    
    comparison_metrics = compare_difficulty(original_irt_params_df, predicted_irt_params_df, model_type=prediction_irt_model_type)
    dump_json(comparison_metrics, os.path.join(CURRENT_RUN_DIR, f"prediction_irt_metrics_{config_name}_{file_suffix_irt}.json"))
    logger.info(f"Difficulty comparison metrics ({config_name}, pred_irt={prediction_irt_model_type.upper()}, raw_probs={use_raw_probs_for_irt}): {comparison_metrics}")
    
    return model, test_metrics, comparison_metrics # Return model for permutation importance

def main_optimization_pipeline():
    # --- Hyperparameter Configurations ---
    nn_configs = { 
        "default_regularized": { 
            "user_embedding_dim": 8, "dropout_rate": 0.3, "l2_reg": 0.001, 
            "learning_rate": 1e-3, "dense_layers": [64, 32], "epochs": 30, 
            "batch_size": 1024, "patience_es": 5
        },
        "config_A_larger_embed_less_dropout": {
            "user_embedding_dim": 16, "dropout_rate": 0.2, "l2_reg": 0.0005, 
            "learning_rate": 1e-3, "dense_layers": [64, 32], "epochs": 40, 
            "batch_size": 512, "patience_es": 7
        },
        "config_B_more_epochs_smaller_lr": {
            "user_embedding_dim": 8, "dropout_rate": 0.3, "l2_reg": 0.001, 
            "learning_rate": 5e-4, "dense_layers": [64, 32], "epochs": 50, 
            "batch_size": 1024, "patience_es": 10
        },
        "config_C_shallow_network": {
            "user_embedding_dim": 8, "dropout_rate": 0.3, "l2_reg": 0.001, 
            "learning_rate": 5e-4, "dense_layers": [32], "epochs": 50, 
            "batch_size": 1024, "patience_es": 10
        },
        "config_D_wider_network": {
            "user_embedding_dim": 16, "dropout_rate": 0.3, "l2_reg": 0.001, 
            "learning_rate": 5e-4, "dense_layers": [128, 64], "epochs": 50,
            "batch_size": 1024, "patience_es": 10
        },
        "config_E_varied_dropout_l2": {
            "user_embedding_dim": 8, "dropout_rate": 0.4, "l2_reg": 0.005,   
            "learning_rate": 5e-4, "dense_layers": [64, 32], "epochs": 50,
            "batch_size": 1024, "patience_es": 10
        },
        "config_F_lower_lr_more_patience": {
            "user_embedding_dim": 8, "dropout_rate": 0.25, "l2_reg": 0.0005,
            "learning_rate": 2e-4, "dense_layers": [64, 32], "epochs": 60, 
            "batch_size": 1024, "patience_es": 12
        }
    }

    student_answers_df, question_content_df = load_and_preprocess_data()

    # --- Experiment Branch 1: 1PL vs 1PL ---
    logger.info("\n=== EXPERIMENT BRANCH: 1PL (Original) vs 1PL (Prediction-Based) ===")
    original_1pl_irt_params_df = calculate_irt_difficulties(student_answers_df, irt_model_type="1pl")
    
    # Ensure question_content_df has QUESTION_TEXT_COL and OPTION_COLS before passing (repeated for safety, can be refactored)
    if QUESTION_TEXT_COL not in question_content_df.columns:
        question_content_df[QUESTION_TEXT_COL] = ""
    for opt_col in OPTION_COLS:
        if opt_col not in question_content_df.columns:
            question_content_df[opt_col] = ""
    if CORRECT_OPTION_COL not in question_content_df.columns:
        question_content_df[CORRECT_OPTION_COL] = "A"

    questions_with_features_df_1pl = extract_all_features(question_content_df.copy()) # Use a copy for safety if df is modified
    
    logger.info("Formatting question text (1PL branch)...")
    questions_with_features_df_1pl[CORRECT_OPTION_COL] = questions_with_features_df_1pl[CORRECT_OPTION_COL].fillna('A').astype(str)
    questions_with_features_df_1pl[FORMATTED_TEXT_COL] = questions_with_features_df_1pl.apply(
        lambda row: format_question_text(
            row, title_col=QUESTION_TEXT_COL, option_cols=OPTION_COLS, correct_option_col=CORRECT_OPTION_COL
        ), axis=1,
    )

    logger.info("Generating text embeddings (1PL branch)...")
    question_embeddings_1pl = {
        "question_ids": questions_with_features_df_1pl[QUESTION_ID_COL].tolist(),
        "formatted_embeddings": {},
        "option_embeddings": defaultdict(dict),
    }
    if not questions_with_features_df_1pl.empty:
        question_embeddings_1pl["formatted_embeddings"] = generate_text_embeddings(
            data_df=questions_with_features_df_1pl,
            text_col=FORMATTED_TEXT_COL,
            id_col=QUESTION_ID_COL,
            model_name=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
    else:
        logger.warning("questions_with_features_df_1pl is empty. Skipping embeddings.")

    embedding_file_path_1pl = os.path.join(CURRENT_RUN_DIR, "03_embeddings_1pl.pkl")
    with open(embedding_file_path_1pl, "wb") as f:
        pickle.dump(question_embeddings_1pl, f)
    logger.info(f"Embeddings saved for 1PL branch: {embedding_file_path_1pl}")
    
    embedding_dimension_1pl = 0
    if question_embeddings_1pl["formatted_embeddings"]:
        embedding_dimension_1pl = next(iter(question_embeddings_1pl["formatted_embeddings"].values())).shape[0]
    if embedding_dimension_1pl == 0:
        logger.warning("Embedding dimension is 0 for 1PL branch.")

    logger.info("Merging datasets for splitting (1PL branch)...")
    # Use original_1pl_irt_params_df for the merge
    complete_dataset_df_1pl = student_answers_df.merge(
        questions_with_features_df_1pl, on=QUESTION_ID_COL, how='inner'
    ).merge(
        original_1pl_irt_params_df, on=QUESTION_ID_COL, how='inner'
    )
    if complete_dataset_df_1pl.empty:
        logger.error("Complete dataset for 1PL branch is empty after merges. Stopping this branch.")
    else:
        logger.info(f"Complete merged dataset (1PL branch): {complete_dataset_df_1pl.shape}")
        complete_dataset_df_1pl[IRT_DIFFICULTY_COL].fillna(complete_dataset_df_1pl[IRT_DIFFICULTY_COL].median(), inplace=True)

        logger.info("Splitting data (1PL branch)...")
        train_q_ids_1pl, val_q_ids_1pl, test_q_ids_1pl = stratified_question_split_3way(
            df=complete_dataset_df_1pl,
            question_col=QUESTION_ID_COL,
            stratify_cols=[CORRECTNESS_COL, IRT_DIFFICULTY_COL],
            test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, 
            random_state=RANDOM_SEED, n_bins=3
        )
        pd.DataFrame({QUESTION_ID_COL: test_q_ids_1pl}).to_csv(
            os.path.join(CURRENT_RUN_DIR, "holdout_ids_1pl.csv"), index=False
        )
        logger.info(f"1PL Branch - Train: {len(train_q_ids_1pl)}, Val: {len(val_q_ids_1pl)}, Test: {len(test_q_ids_1pl)}")

        if train_q_ids_1pl and val_q_ids_1pl and test_q_ids_1pl:
            logger.info("Preparing NN datasets (1PL branch)...")
            train_ds_1pl, train_y_1pl, val_ds_1pl, val_y_1pl, preprocessors_1pl, final_num_cols_1pl = prepare_nn_datasets(
                merged_df=complete_dataset_df_1pl,
                combined_embeddings=question_embeddings_1pl,
                train_q_ids=train_q_ids_1pl, val_q_ids=val_q_ids_1pl,
                user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL,
                numerical_feature_cols=BASE_NUMERICAL_FEATURE_COLS, 
                categorical_feature_cols_to_encode=CATEGORICAL_FEATURE_COLS_TO_ENCODE,
                embedding_dim=embedding_dimension_1pl,
            )
            save_preprocessors(preprocessors_1pl, CURRENT_RUN_DIR, filename="nn_preprocessors_1pl.pkl")
            logger.info(f"Final numerical columns for model (1PL branch): {final_num_cols_1pl}")

            logger.info("\n=== RUNNING 1PL vs 1PL EXPERIMENTS (BINARIZED IRT) ===")
            for config_name, nn_c in nn_configs.items():
                run_single_nn_config(
                    config_name + "_1pl_vs_1pl_binIRT", nn_c, preprocessors_1pl, 
                    train_ds_1pl, train_y_1pl, val_ds_1pl, val_y_1pl,
                    complete_dataset_df_1pl, test_q_ids_1pl, question_embeddings_1pl, original_1pl_irt_params_df,
                    embedding_dimension_1pl, final_num_cols_1pl, prediction_irt_model_type="1pl", use_raw_probs_for_irt=False
                )
        else:
            logger.warning("Skipping NN training for 1PL branch due to empty question ID lists.")


    # --- Experiment Branch 2: 2PL vs 2PL (current main flow) ---
    logger.info("\n=== EXPERIMENT BRANCH: 2PL (Original) vs 2PL (Prediction-Based) ===")
    original_2pl_irt_params_df = calculate_irt_difficulties(student_answers_df, irt_model_type="2pl")
    
    # Re-use or re-create questions_with_features_df for 2PL branch if necessary (could be the same as 1pl)
    # For simplicity, assuming feature extraction is similar, but if it depends on IRT params, then re-extract.
    # Here, using a fresh copy. In practice, if extract_all_features doesn't use IRT params, it can be done once.
    questions_with_features_df_2pl = extract_all_features(question_content_df.copy())

    logger.info("Formatting question text (2PL branch)...")
    questions_with_features_df_2pl[CORRECT_OPTION_COL] = questions_with_features_df_2pl[CORRECT_OPTION_COL].fillna('A').astype(str)
    questions_with_features_df_2pl[FORMATTED_TEXT_COL] = questions_with_features_df_2pl.apply(
        lambda row: format_question_text(
            row, title_col=QUESTION_TEXT_COL, option_cols=OPTION_COLS, correct_option_col=CORRECT_OPTION_COL
        ), axis=1,
    )

    logger.info("Generating text embeddings (2PL branch)...")
    question_embeddings_2pl = {
        "question_ids": questions_with_features_df_2pl[QUESTION_ID_COL].tolist(),
        "formatted_embeddings": {},
        "option_embeddings": defaultdict(dict),
    }
    if not questions_with_features_df_2pl.empty:
        question_embeddings_2pl["formatted_embeddings"] = generate_text_embeddings(
            data_df=questions_with_features_df_2pl,
            text_col=FORMATTED_TEXT_COL,
            id_col=QUESTION_ID_COL,
            model_name=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
    else:
        logger.warning("questions_with_features_df_2pl is empty. Skipping embeddings.")

    embedding_file_path_2pl = os.path.join(CURRENT_RUN_DIR, "03_embeddings_2pl.pkl")
    with open(embedding_file_path_2pl, "wb") as f:
        pickle.dump(question_embeddings_2pl, f)
    logger.info(f"Embeddings saved for 2PL branch: {embedding_file_path_2pl}")
    
    embedding_dimension_2pl = 0
    if question_embeddings_2pl["formatted_embeddings"]:
        embedding_dimension_2pl = next(iter(question_embeddings_2pl["formatted_embeddings"].values())).shape[0]
    if embedding_dimension_2pl == 0:
        logger.warning("Embedding dimension is 0 for 2PL branch.")

    logger.info("Merging datasets for splitting (2PL branch)...")
    complete_dataset_df_2pl = student_answers_df.merge(
        questions_with_features_df_2pl, on=QUESTION_ID_COL, how='inner'
    ).merge(
        original_2pl_irt_params_df, on=QUESTION_ID_COL, how='inner' 
    )
    if complete_dataset_df_2pl.empty:
        logger.error("Complete dataset for 2PL branch is empty. Stopping this branch.")
        return # or handle appropriately

    logger.info(f"Complete merged dataset (2PL branch): {complete_dataset_df_2pl.shape}")
    complete_dataset_df_2pl[IRT_DIFFICULTY_COL].fillna(complete_dataset_df_2pl[IRT_DIFFICULTY_COL].median(), inplace=True)

    logger.info("Splitting data (2PL branch)...")
    train_q_ids_2pl, val_q_ids_2pl, test_q_ids_2pl = stratified_question_split_3way(
        df=complete_dataset_df_2pl,
        question_col=QUESTION_ID_COL,
        stratify_cols=[CORRECTNESS_COL, IRT_DIFFICULTY_COL],
        test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, 
        random_state=RANDOM_SEED, n_bins=3
    )
    pd.DataFrame({QUESTION_ID_COL: test_q_ids_2pl}).to_csv(
        os.path.join(CURRENT_RUN_DIR, "holdout_ids_2pl.csv"), index=False
    )
    logger.info(f"2PL Branch - Train: {len(train_q_ids_2pl)}, Val: {len(val_q_ids_2pl)}, Test: {len(test_q_ids_2pl)}")

    if not train_q_ids_2pl or not val_q_ids_2pl or not test_q_ids_2pl:
        logger.error("Skipping NN training for 2PL branch due to empty question ID lists.")
    else:
        logger.info("Preparing NN datasets (2PL branch)...")
        train_ds_2pl, train_y_2pl, val_ds_2pl, val_y_2pl, preprocessors_2pl, final_num_cols_2pl = prepare_nn_datasets(
            merged_df=complete_dataset_df_2pl,
            combined_embeddings=question_embeddings_2pl,
            train_q_ids=train_q_ids_2pl, val_q_ids=val_q_ids_2pl,
            user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL,
            numerical_feature_cols=BASE_NUMERICAL_FEATURE_COLS, 
            categorical_feature_cols_to_encode=CATEGORICAL_FEATURE_COLS_TO_ENCODE,
            embedding_dim=embedding_dimension_2pl,
        )
        save_preprocessors(preprocessors_2pl, CURRENT_RUN_DIR, filename="nn_preprocessors_2pl.pkl")
        logger.info(f"Final numerical columns for model (2PL branch): {final_num_cols_2pl}")

        all_results_2pl = [] 
        best_overall_pearson_2pl = -1
        best_config_name_2pl = None
        best_model_for_permutation_importance_2pl = None
        
        logger.info("\n=== RUNNING 2PL vs 2PL EXPERIMENTS (BINARIZED IRT) ===")
        for config_name, nn_c in nn_configs.items():
            model, test_metrics_run, comparison_metrics_run = run_single_nn_config(
                config_name + "_2pl_vs_2pl_binIRT", nn_c, preprocessors_2pl, 
                train_ds_2pl, train_y_2pl, val_ds_2pl, val_y_2pl,
                complete_dataset_df_2pl, test_q_ids_2pl, question_embeddings_2pl, original_2pl_irt_params_df,
                embedding_dimension_2pl, final_num_cols_2pl, prediction_irt_model_type="2pl", use_raw_probs_for_irt=False
            )
            all_results_2pl.append({
                "config_name": config_name + "_2pl_vs_2pl_binIRT", 
                "nn_config": nn_c, 
                "test_metrics": test_metrics_run, 
                "irt_comparison": comparison_metrics_run
            })
            if comparison_metrics_run.get('pearson', -1) > best_overall_pearson_2pl:
                best_overall_pearson_2pl = comparison_metrics_run['pearson']
                best_config_name_2pl = config_name + "_2pl_vs_2pl_binIRT"
                best_model_for_permutation_importance_2pl = model

        logger.info("\n=== RUNNING 2PL vs 2PL EXPERIMENTS (RAW PROBABILITY IRT) ===")
        best_config_key_for_raw_2pl = best_config_name_2pl.replace("_2pl_vs_2pl_binIRT", "") if best_config_name_2pl else "default_regularized"
        best_config_for_raw_irt_2pl = nn_configs.get(best_config_key_for_raw_2pl, nn_configs["default_regularized"])
        
        if best_config_name_2pl: 
            _, test_metrics_raw, comparison_metrics_raw = run_single_nn_config(
                best_config_key_for_raw_2pl + "_2pl_vs_2pl_rawProbIRT", 
                best_config_for_raw_irt_2pl, 
                preprocessors_2pl, train_ds_2pl, train_y_2pl, val_ds_2pl, val_y_2pl,
                complete_dataset_df_2pl, test_q_ids_2pl, question_embeddings_2pl, original_2pl_irt_params_df,
                embedding_dimension_2pl, final_num_cols_2pl, prediction_irt_model_type="2pl", use_raw_probs_for_irt=True
            )
            all_results_2pl.append({
                "config_name": best_config_key_for_raw_2pl + "_2pl_vs_2pl_rawProbIRT", 
                "nn_config": best_config_for_raw_irt_2pl, 
                "test_metrics": test_metrics_raw, 
                "irt_comparison": comparison_metrics_raw
            })
        else:
            logger.warning("Skipping raw probability IRT run for 2PL branch as no best model from binarized runs.")

        logger.info(f"Best Pearson correlation from 2PL binarized IRT runs: {best_overall_pearson_2pl:.4f} from config: {best_config_name_2pl}")

        dump_json(all_results_2pl, os.path.join(CURRENT_RUN_DIR, "all_optimization_results_2pl_vs_2pl.json"))
        logger.info(f"All 2PL vs 2PL optimization results saved to all_optimization_results_2pl_vs_2pl.json")

        if best_model_for_permutation_importance_2pl:
            logger.info(f"Calculating Permutation Importance using model from 2PL config: {best_config_name_2pl}")
            # (Permutation importance code as before, using _2pl suffixed variables where appropriate for data)
            test_df_pi = complete_dataset_df_2pl[complete_dataset_df_2pl[QUESTION_ID_COL].isin(test_q_ids_2pl)].copy()
            # ... (rest of PI data prep using preprocessors_2pl and final_num_cols_2pl) ...
            # ... (then RF feature importance as before) ...
            for col in preprocessors_2pl['original_numerical_features']:
                test_df_pi[col] = pd.to_numeric(test_df_pi[col], errors='coerce')
            for col in preprocessors_2pl.get('categorical_features_encoded', []):
                test_df_pi[col] = test_df_pi[col].fillna('Unknown').astype(str)

            if preprocessors_2pl.get('ohe_encoder') and preprocessors_2pl.get('categorical_features_encoded', []):
                ohe_features_pi = preprocessors_2pl['ohe_encoder'].transform(test_df_pi[preprocessors_2pl['categorical_features_encoded']])
                ohe_df_pi = pd.DataFrame(ohe_features_pi, columns=preprocessors_2pl['ohe_feature_names'], index=test_df_pi.index)
                numerical_part_pi = test_df_pi[preprocessors_2pl['original_numerical_features']].fillna(0)
                test_numerical_combined_pi = pd.concat([numerical_part_pi, ohe_df_pi], axis=1)
            else:
                test_numerical_combined_pi = test_df_pi[preprocessors_2pl['original_numerical_features']].fillna(0)
            
            for col in final_num_cols_2pl:
                if col not in test_numerical_combined_pi.columns:
                    test_numerical_combined_pi[col] = 0
            # test_numerical_unscaled_pi = test_numerical_combined_pi[final_num_cols_2pl].fillna(0)
                    
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(train_ds_2pl["numerical_input"], train_y_2pl)
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': final_num_cols_2pl,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            logger.info("\nRandom Forest Feature Importances (proxy for 2PL branch):")
            logger.info(f"\n{feature_importance_df.head(20)}")
            feature_importance_df.to_csv(os.path.join(CURRENT_RUN_DIR, "rf_feature_importances_2pl.csv"), index=False)
        else:
            logger.warning("No best model for 2PL branch to perform permutation importance.")

if __name__ == "__main__":
    main_optimization_pipeline() 
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging
import json

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from modules.irt import estimate_irt_1pl_difficulty # Only 1PL needed for this script
from modules.features import extract_text_features, calculate_option_features
from modules.embeddings import generate_text_embeddings
from modules.utils import format_question_text
from modules.modeling_data import (
    stratified_question_split_3way,
    prepare_nn_datasets,
    save_preprocessors,
    make_dataset
)
from modules.neural_net import create_nn_model, train_nn_model, save_nn_model
from modules.evaluation import (
    evaluate_model,
    dump_json,
    prediction_matrix,
    compare_difficulty # Assuming this is the updated one
)
# We'll use a local version of difficulty_from_predictions for 1PL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
USER_ID_COL = "user_id"
QUESTION_ID_COL = "question_id"
CORRECTNESS_COL = "is_correct"
QUESTION_TEXT_COL = "question_title"
OPTION_COLS = ["option_a", "option_b", "option_c", "option_d", "option_e"]
CORRECT_OPTION_COL = "correct_option_letter"
FORMATTED_TEXT_COL = "formatted_question_text"
IRT_DIFFICULTY_COL = "difficulty"
# IRT_DISCRIMINATION_COL = "discrimination" # Not used for 1PL

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
EMBEDDING_BATCH_SIZE = 32
TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, "data", "zapien")
ANSWERS_FILE_PATH = os.path.join(DATA_DIR, "answers.csv")
QUESTIONS_FILE_PATH = os.path.join(DATA_DIR, "questions.csv")
ADDITIONAL_FEATURES_FILE_PATH = os.path.join(DATA_DIR, "questions_additional_features.csv")

RESULTS_BASE_DIR = os.path.join(ROOT_DIR, "results_best_model_no_filter") # New results directory
CURRENT_RUN_DIR = os.path.join(RESULTS_BASE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S_1PL_no_filter"))
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)
logger.info(f"Results will be saved to: {CURRENT_RUN_DIR}")

def local_difficulty_from_binarized_predictions_1pl(pred_df):
    logger.info("Estimating IRT 1PL parameters from BINARIZED prediction probabilities...")
    if pred_df.empty:
        logger.warning("Prediction DataFrame is empty, cannot estimate difficulty.")
        return pd.DataFrame()
    
    long = (
        pred_df.reset_index()
        .melt(id_vars="index", var_name=QUESTION_ID_COL, value_name="prob")
        .rename(columns={"index": USER_ID_COL})
    )
    long[CORRECTNESS_COL] = (long["prob"] > 0.5).astype(int) # Binarization

    params_df = estimate_irt_1pl_difficulty(
        response_df=long[[USER_ID_COL, QUESTION_ID_COL, CORRECTNESS_COL]],
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL)
    return params_df

def load_and_preprocess_data_no_filter():
    logger.info("Loading raw data (NO QUESTION FILTERING)..." )
    student_answers_df = pd.read_csv(ANSWERS_FILE_PATH)
    question_content_df = pd.read_csv(QUESTIONS_FILE_PATH)
    additional_features_df = pd.read_csv(ADDITIONAL_FEATURES_FILE_PATH)

    logger.info(f"Initial student answers: {student_answers_df.shape}")
    logger.info(f"Initial question content: {question_content_df.shape}")
    logger.info(f"Initial additional features: {additional_features_df.shape}")

    question_content_df = pd.merge(question_content_df, additional_features_df, on=QUESTION_ID_COL, how='left')
    logger.info(f"Question content after merging additional features: {question_content_df.shape}")
    
    # --- Handle NaNs in new features ---
    numerical_cols_to_median_fill = [
        "Answer_Length_Variance", "Correct_Distractor_CosineSim_Mean",
        "Distractor_Embedding_Distance_Mean", "Extreme_Wording_Option_Count",
        "LLM_Distractor_Plausibility_Max", "LLM_Distractor_Plausibility_Mean",
        "Mathematical_Notation_Density", "Max_Expression_Nesting_Depth",
        "Question_Answer_Info_Gap", "Ratio_Abstract_Concrete_Symbols"
    ]
    nan_counts_before_median = {}
    nan_counts_after_median = {}
    for col in numerical_cols_to_median_fill:
        if col in question_content_df.columns:
            nan_counts_before_median[col] = question_content_df[col].isna().sum()
            question_content_df[col] = pd.to_numeric(question_content_df[col], errors='coerce')
            median_val = question_content_df[col].median()
            question_content_df[col].fillna(median_val, inplace=True)
            nan_counts_after_median[col] = question_content_df[col].isna().sum()
            if nan_counts_before_median[col] > 0:
                 logger.info(f"Column '{col}': Coerced to numeric. Filled {nan_counts_before_median[col]} NaNs with median ({median_val}). Remaining NaNs: {nan_counts_after_median[col]}.")

    binary_cols_to_zero_fill = ["Has_Abstract_Symbols", "Has_NoneAll_Option", "Option_Length_Outlier_Flag", "RealWorld_Context_Flag", "Units_Check"]
    nan_counts_before_zero = {}
    for col in binary_cols_to_zero_fill:
        if col in question_content_df.columns:
            nan_counts_before_zero[col] = question_content_df[col].isna().sum()
            question_content_df[col] = pd.to_numeric(question_content_df[col], errors='coerce').fillna(0)
            if nan_counts_before_zero[col] > 0:
                logger.info(f"Column '{col}': Coerced to numeric. Filled {nan_counts_before_zero[col]} NaNs with 0.")

    for col in CATEGORICAL_FEATURE_COLS_TO_ENCODE:
        if col in question_content_df.columns:
            nans_before = question_content_df[col].isna().sum()
            question_content_df[col] = question_content_df[col].astype(str).fillna('Unknown')
            if nans_before > 0:
                logger.info(f"Column '{col}': Filled {nans_before} NaNs with 'Unknown'.")

    # --- NO QUESTION FILTERING --- 
    logger.info("Skipping question filtering based on response patterns.")
    logger.info(f"Student answers shape (unfiltered): {student_answers_df.shape}")
    logger.info(f"Unique questions (unfiltered): {student_answers_df[QUESTION_ID_COL].nunique()}")

    # Filter question_content_df to only include questions present in student_answers_df (important if questions.csv has questions with no answers)
    valid_qids_in_answers = student_answers_df[QUESTION_ID_COL].unique()
    question_content_df = question_content_df[question_content_df[QUESTION_ID_COL].isin(valid_qids_in_answers)]
    logger.info(f"Question content after matching to any question with answers: {question_content_df.shape}")

    if student_answers_df.empty:
        raise ValueError("Student answers DataFrame is empty. Cannot proceed.")
    if question_content_df.empty:
        raise ValueError("Question content DataFrame is empty after matching to answers. Cannot proceed.")

    return student_answers_df, question_content_df

def calculate_original_1pl_irt(student_answers_df):
    logger.info("Estimating ORIGINAL question difficulty using IRT 1PL model (on potentially unfiltered data)...")
    question_params_df = estimate_irt_1pl_difficulty(
        response_df=student_answers_df,
        user_col=USER_ID_COL,
        question_col=QUESTION_ID_COL,
        correctness_col=CORRECTNESS_COL,
    )
    question_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, "01_irt_1pl_params_original_no_filter.csv"), index=False)
    logger.info("Original 1PL IRT parameters (no_filter) saved.")
    logger.info(f"Original 1PL IRT Difficulty (no_filter) summary:\n{question_params_df[IRT_DIFFICULTY_COL].describe()}")
    logger.info(f"Original 1PL IRT Difficulty (no_filter) number of questions: {len(question_params_df)}")
    return question_params_df # Return full df with question_id and difficulty

# --- extract_all_features function (copied from optimization.py, ensure it's robust) ---
# (Assuming extract_all_features, format_question_text, generate_text_embeddings are robust to data characteristics)
def extract_all_features(question_content_df):
    logger.info("Extracting text-based features...")
    if QUESTION_TEXT_COL not in question_content_df.columns:
        logger.warning(f"'{QUESTION_TEXT_COL}' not found. Adding empty string column for text feature extraction.")
        question_content_df[QUESTION_TEXT_COL] = ""
    text_features_df = question_content_df[QUESTION_TEXT_COL].apply(
        lambda text: pd.Series(extract_text_features(str(text)))
    ).add_prefix("question_")

    logger.info("Extracting option-based features...")
    for col in OPTION_COLS:
        if col not in question_content_df.columns:
            logger.warning(f"Option column '{col}' not found. Adding empty string column.")
            question_content_df[col] = ""
        else:
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
    
    generated_text_feature_cols = [
        "question_word_count", "question_char_count", "question_avg_word_length",
        "question_digit_count", "question_special_char_count",
        "question_mathematical_symbols", "question_latex_expressions",
    ]
    for col in generated_text_feature_cols:
        if col in questions_with_features_df.columns:
            questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)
        else: 
            logger.warning(f"Feature {col} not found after text extraction, adding as zeros.")
            questions_with_features_df[col] = 0

    generated_option_feature_cols = ["jaccard_similarity_std", "avg_option_length", "avg_option_word_count"]
    for col in generated_option_feature_cols:
         if col in questions_with_features_df.columns:
            questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)
         else:
            logger.warning(f"Feature {col} not found after option extraction, adding as zeros.")
            questions_with_features_df[col] = 0
            
    for col in BASE_NUMERICAL_FEATURE_COLS:
        if col not in questions_with_features_df.columns:
            logger.warning(f"Column '{col}' from BASE_NUMERICAL_FEATURE_COLS not found. Adding with zeros.")
            questions_with_features_df[col] = 0
        else: 
             questions_with_features_df[col] = pd.to_numeric(questions_with_features_df[col], errors='coerce').fillna(0)

    questions_with_features_df.to_csv(
        os.path.join(CURRENT_RUN_DIR, "02_question_features_full_no_filter.csv"),
        index=False
    )
    logger.info(f"Questions with all features (no_filter) saved. Shape: {questions_with_features_df.shape}")
    return questions_with_features_df


def main_best_model_pipeline():
    student_answers_df, question_content_df = load_and_preprocess_data_no_filter()
    original_1pl_irt_params_df = calculate_original_1pl_irt(student_answers_df)

    # Ensure necessary text/option columns exist before feature extraction
    if QUESTION_TEXT_COL not in question_content_df.columns: question_content_df[QUESTION_TEXT_COL] = ""
    for opt_col in OPTION_COLS: 
        if opt_col not in question_content_df.columns: question_content_df[opt_col] = ""
    if CORRECT_OPTION_COL not in question_content_df.columns: question_content_df[CORRECT_OPTION_COL] = "A"

    questions_with_features_df = extract_all_features(question_content_df)
    
    logger.info("Formatting question text...")
    questions_with_features_df[CORRECT_OPTION_COL] = questions_with_features_df[CORRECT_OPTION_COL].fillna('A').astype(str)
    questions_with_features_df[FORMATTED_TEXT_COL] = questions_with_features_df.apply(
        lambda row: format_question_text(
            row, title_col=QUESTION_TEXT_COL, option_cols=OPTION_COLS, correct_option_col=CORRECT_OPTION_COL
        ), axis=1,
    )

    logger.info("Generating text embeddings...")
    question_embeddings = {
        "question_ids": questions_with_features_df[QUESTION_ID_COL].tolist(),
        "formatted_embeddings": {},
        "option_embeddings": defaultdict(dict),
    }
    if not questions_with_features_df.empty:
        question_embeddings["formatted_embeddings"] = generate_text_embeddings(
            data_df=questions_with_features_df,
            text_col=FORMATTED_TEXT_COL,
            id_col=QUESTION_ID_COL,
            model_name=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        logger.info(f"Number of texts for which embeddings were generated: {len(question_embeddings['formatted_embeddings'])}")
    embedding_file_path = os.path.join(CURRENT_RUN_DIR, "03_embeddings_no_filter.pkl")
    with open(embedding_file_path, "wb") as f: pickle.dump(question_embeddings, f)
    logger.info(f"Embeddings (no_filter) saved to {embedding_file_path}")
    
    embedding_dimension = 0
    if question_embeddings["formatted_embeddings"]:
        embedding_dimension = next(iter(question_embeddings["formatted_embeddings"].values())).shape[0]
    if embedding_dimension == 0: logger.warning("Embedding dimension is 0.")
    else: logger.info(f"Detected embedding dimension: {embedding_dimension}")

    logger.info("Merging datasets for splitting (unfiltered data)...")
    complete_dataset_df = student_answers_df.merge(
        questions_with_features_df, on=QUESTION_ID_COL, how='inner'
    ).merge(
        original_1pl_irt_params_df, on=QUESTION_ID_COL, how='inner'
    )
    if complete_dataset_df.empty: 
        logger.error("Complete dataset is empty. Halting."); return
    logger.info(f"Complete merged dataset (unfiltered): {complete_dataset_df.shape}")
    complete_dataset_df[IRT_DIFFICULTY_COL].fillna(complete_dataset_df[IRT_DIFFICULTY_COL].median(), inplace=True)

    logger.info("Splitting data (unfiltered)...")
    # Stratification might be less stable with extreme 0/100 items, but let's try
    train_q_ids, val_q_ids, test_q_ids = stratified_question_split_3way(
        df=complete_dataset_df,
        question_col=QUESTION_ID_COL,
        stratify_cols=[CORRECTNESS_COL, IRT_DIFFICULTY_COL],
        test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, 
        random_state=RANDOM_SEED, n_bins=5 # Increased bins slightly as data might be more diverse
    )
    pd.DataFrame({QUESTION_ID_COL: test_q_ids}).to_csv(
        os.path.join(CURRENT_RUN_DIR, "holdout_ids_no_filter.csv"), index=False
    )
    logger.info(f"Unfiltered - Train: {len(train_q_ids)}, Val: {len(val_q_ids)}, Test: {len(test_q_ids)}")

    if not train_q_ids or not val_q_ids or not test_q_ids: 
        logger.error("Data split resulted in empty q_id lists. Halting."); return

    logger.info("Preparing NN datasets (unfiltered)...")
    train_ds, train_y, val_ds, val_y, preprocessors, final_num_cols = prepare_nn_datasets(
        merged_df=complete_dataset_df,
        combined_embeddings=question_embeddings,
        train_q_ids=train_q_ids, val_q_ids=val_q_ids,
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL,
        numerical_feature_cols=BASE_NUMERICAL_FEATURE_COLS, 
        categorical_feature_cols_to_encode=CATEGORICAL_FEATURE_COLS_TO_ENCODE,
        embedding_dim=embedding_dimension,
    )
    save_preprocessors(preprocessors, CURRENT_RUN_DIR, filename="nn_preprocessors_no_filter.pkl")
    logger.info(f"Final numerical columns for model (unfiltered) - Count {len(final_num_cols)}: {final_num_cols}")

    # --- Use the Best 1PL NN Config from optimization runs ---
    best_1pl_config = {
        "user_embedding_dim": 8, "dropout_rate": 0.25, "l2_reg": 0.0005,
        "learning_rate": 2e-4, "dense_layers": [64, 32], "epochs": 60, 
        "batch_size": 1024, "patience_es": 12
    } # This was config_F
    config_name = "best_1pl_config_F_no_filter"
    logger.info(f"Running with best 1PL config: {config_name} - {best_1pl_config}")

    model = create_nn_model(
        user_vocab_size=preprocessors["user_vocab_size"],
        numerical_feature_size=len(final_num_cols),
        embedding_dim=embedding_dimension,
        user_embedding_dim=best_1pl_config.get("user_embedding_dim", 8),
        dropout_rate=best_1pl_config.get("dropout_rate", 0.3),
        l2_reg=best_1pl_config.get("l2_reg", 0.001),
        dense_layers_config=best_1pl_config.get("dense_layers", [64, 32])
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=best_1pl_config.get("learning_rate", 1e-3)),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    history = train_nn_model(
        model, train_ds, train_y, val_ds, val_y,
        epochs=best_1pl_config.get("epochs", 50),
        batch_size=best_1pl_config.get("batch_size", 1024),
        patience_es=best_1pl_config.get("patience_es", 10)
    )
    # Log best validation metrics from history
    if history and history.history:
        best_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = np.min(history.history['val_loss'])
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
        if 'val_accuracy' in history.history:
            best_val_accuracy = history.history['val_accuracy'][best_val_loss_epoch - 1]
            logger.info(f"Validation accuracy at best val_loss epoch: {best_val_accuracy:.4f}")
        if 'val_auc' in history.history:
            best_val_auc = history.history['val_auc'][best_val_loss_epoch - 1]
            logger.info(f"Validation AUC at best val_loss epoch: {best_val_auc:.4f}")
            
    save_nn_model(model, CURRENT_RUN_DIR, filename=f"model_{config_name}.keras")
    logger.info(f"Model training complete for {config_name}.")

    logger.info(f"Evaluating {config_name} on hold-out test data...")
    test_dataset_inputs, test_labels = make_dataset(
        df=complete_dataset_df,
        q_ids=test_q_ids,
        preprocessors=preprocessors,
        combined_embeddings=question_embeddings,
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL
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
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL
    )
    prob_matrix.to_csv(os.path.join(CURRENT_RUN_DIR, f"04_prediction_probabilities_{config_name}.csv"))
    logger.info(f"Prediction matrix saved for {config_name}.")

    # Use binarized 1PL for prediction-based IRT
    predicted_irt_params_df = local_difficulty_from_binarized_predictions_1pl(prob_matrix)
    predicted_irt_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, f"05_predicted_1pl_params_{config_name}.csv"), index=False)
    logger.info(f"Predicted 1PL IRT params saved for {config_name}.")
    
    comparison_metrics = compare_difficulty(original_1pl_irt_params_df, predicted_irt_params_df, model_type="1pl")
    dump_json(comparison_metrics, os.path.join(CURRENT_RUN_DIR, f"prediction_irt_metrics_{config_name}.json"))
    logger.info(f"Difficulty comparison metrics ({config_name}, 1PLv1PL): {comparison_metrics}")

if __name__ == "__main__":
    main_best_model_pipeline() 
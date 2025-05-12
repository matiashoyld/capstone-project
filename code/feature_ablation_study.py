import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging
import json
import sys

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --- Add Modules to Path ---
ROOT_DIR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODULES_PATH = os.path.join(ROOT_DIR_SCRIPT, 'code', 'modules')
if MODULES_PATH not in sys.path:
    sys.path.insert(0, MODULES_PATH)

from modules.irt import estimate_irt_1pl_difficulty
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
    compare_difficulty
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Constants ---
USER_ID_COL = "user_id"
QUESTION_ID_COL = "question_id"
CORRECTNESS_COL = "is_correct"
QUESTION_TEXT_COL = "question_title"
OPTION_COLS = ["option_a", "option_b", "option_c", "option_d", "option_e"]
CORRECT_OPTION_COL = "correct_option_letter"
FORMATTED_TEXT_COL = "formatted_question_text"
IRT_DIFFICULTY_COL = "difficulty" # Only for stratification

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
EMBEDDING_BATCH_SIZE = 32
TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

# --- Path Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, "data", "zapien")
ANSWERS_FILE_PATH = os.path.join(DATA_DIR, "answers.csv")
QUESTIONS_FILE_PATH = os.path.join(DATA_DIR, "questions.csv")
ADDITIONAL_FEATURES_FILE_PATH = os.path.join(DATA_DIR, "questions_additional_features.csv")

RESULTS_BASE_DIR = os.path.join(ROOT_DIR, "results_feature_ablation_new_buckets_strategy")
CURRENT_RUN_DIR = os.path.join(RESULTS_BASE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S_ablation_study_new_buckets"))
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)
logger.info(f"Results will be saved to: {CURRENT_RUN_DIR}")

# --- New Feature Buckets Definitions ---
QUESTION_FEATURES_NUMERICAL = [
    "question_word_count", "question_char_count", "question_avg_word_length",
    "question_digit_count", "question_special_char_count",
    "question_mathematical_symbols", "question_latex_expressions",
    "has_image", "Has_Abstract_Symbols"
]
QUESTION_FEATURES_CATEGORICAL = [] # No categorical question features in this definition

OPTION_FEATURES_NUMERICAL = [
    "jaccard_similarity_std", "avg_option_length", "avg_option_word_count",
    "Has_NoneAll_Option", "Answer_Length_Variance", "Option_Length_Outlier_Flag",
    "Correct_Distractor_CosineSim_Mean", 
    "Distractor_Embedding_Distance_Mean"
]
OPTION_FEATURES_CATEGORICAL = [] # No categorical option features

LLM_DERIVED_NUMERICAL = [
    "Mathematical_Notation_Density", "Max_Expression_Nesting_Depth",
    "Ratio_Abstract_Concrete_Symbols", "Units_Check", "avg_steps", "level",
    "num_misconceptions", "Extreme_Wording_Option_Count",
    "LLM_Distractor_Plausibility_Max", "LLM_Distractor_Plausibility_Mean",
    "Question_Answer_Info_Gap", "RealWorld_Context_Flag"
]
LLM_DERIVED_CATEGORICAL = [
    "Most_Complex_Number_Type", "Knowledge_Dimension", "Problem_Archetype"
]

# --- Re-define local_difficulty_from_binarized_predictions_1pl (from run_best_model_no_filter.py) ---
def local_difficulty_from_binarized_predictions_1pl(pred_df):
    logger.info("Estimating IRT 1PL parameters from BINARIZED prediction probabilities...")
    if pred_df.empty:
        logger.warning("Prediction DataFrame is empty, cannot estimate 1PL difficulty.")
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
    question_content_df = pd.merge(question_content_df, additional_features_df, on=QUESTION_ID_COL, how='left')

    # Consolidate all potentially used features for initial NaN handling
    all_numerical_to_coerce = list(set(
        QUESTION_FEATURES_NUMERICAL + OPTION_FEATURES_NUMERICAL + LLM_DERIVED_NUMERICAL
    ))
    all_categorical_to_fill = list(set(
        QUESTION_FEATURES_CATEGORICAL + OPTION_FEATURES_CATEGORICAL + LLM_DERIVED_CATEGORICAL
    ))

    for col in all_numerical_to_coerce:
        if col in question_content_df.columns:
            is_binary_like = col in ["has_image", "Has_Abstract_Symbols", "Has_NoneAll_Option", 
                                  "Option_Length_Outlier_Flag", "Units_Check", "RealWorld_Context_Flag"]
            question_content_df[col] = pd.to_numeric(question_content_df[col], errors='coerce')
            if is_binary_like:
                question_content_df[col].fillna(0, inplace=True)
            else:
                question_content_df[col].fillna(question_content_df[col].median(), inplace=True)
    
    for col in all_categorical_to_fill:
        if col in question_content_df.columns:
             question_content_df[col] = question_content_df[col].astype(str).fillna('Unknown')

    logger.info("Skipping question filtering based on response patterns for ablation study.")
    valid_qids_in_answers = student_answers_df[QUESTION_ID_COL].unique()
    question_content_df = question_content_df[question_content_df[QUESTION_ID_COL].isin(valid_qids_in_answers)]
    
    if student_answers_df.empty: raise ValueError("Student answers DF empty.")
    if question_content_df.empty: raise ValueError("Question content DF empty after matching to answers.")
    return student_answers_df, question_content_df

def extract_selected_features(question_content_df, numerical_to_select, categorical_to_select):
    logger.info("Extracting text-based and option-based features (as needed)...")
    # Ensure QUESTION_TEXT_COL and OPTION_COLS exist if lexical features are needed
    if any(f in QUESTION_FEATURES_NUMERICAL or f in OPTION_FEATURES_NUMERICAL for f in numerical_to_select):
        if QUESTION_TEXT_COL not in question_content_df.columns: question_content_df[QUESTION_TEXT_COL] = ""
        text_features_df = question_content_df[QUESTION_TEXT_COL].apply(
            lambda text: pd.Series(extract_text_features(str(text)))).add_prefix("question_")
        question_content_df = pd.concat([question_content_df, text_features_df], axis=1)

        for col in OPTION_COLS:
            if col not in question_content_df.columns: question_content_df[col] = ""
            else: question_content_df[col] = question_content_df[col].fillna('').astype(str)
        option_features_list = question_content_df.apply(
            lambda row: calculate_option_features(row, OPTION_COLS), axis=1).tolist()
        option_features_df = pd.DataFrame(option_features_list, index=question_content_df.index)
        question_content_df = pd.concat([question_content_df, option_features_df], axis=1)

    # Ensure all selected columns exist, filling with defaults if necessary after extraction
    final_df = pd.DataFrame(index=question_content_df.index)
    final_df[QUESTION_ID_COL] = question_content_df[QUESTION_ID_COL]
    if QUESTION_TEXT_COL in question_content_df.columns : final_df[QUESTION_TEXT_COL] = question_content_df[QUESTION_TEXT_COL]
    if CORRECT_OPTION_COL in question_content_df.columns: final_df[CORRECT_OPTION_COL] = question_content_df[CORRECT_OPTION_COL]
    for opt_col in OPTION_COLS: 
        if opt_col in question_content_df.columns: final_df[opt_col] = question_content_df[opt_col]

    for col in numerical_to_select:
        if col in question_content_df.columns:
            final_df[col] = pd.to_numeric(question_content_df[col], errors='coerce').fillna(0)
        else:
            logger.warning(f"Numerical column '{col}' selected but not found/generated. Adding as zeros.")
            final_df[col] = 0
    
    for col in categorical_to_select:
        if col in question_content_df.columns:
            final_df[col] = question_content_df[col].astype(str).fillna('Unknown')
        else:
            logger.warning(f"Categorical column '{col}' selected but not found/generated. Adding as 'Unknown'.")
            final_df[col] = 'Unknown'
            
    logger.info(f"DataFrame with selected features prepared. Shape: {final_df.shape}")
    return final_df

def run_ablation_experiment(
    exp_config_name, 
    numerical_cols_for_run, categorical_cols_for_run,
    student_answers_df, question_content_df_base, 
    original_1pl_irt_params_df # For stratification and final comparison
    ):
    
    logger.info(f"""\n======================================================================
    RUNNING ABLATION CONFIGURATION: {exp_config_name}
    Numerical Features Used: {len(numerical_cols_for_run)}
    Categorical Features OHE: {len(categorical_cols_for_run)}
======================================================================""")

    current_question_content_df = question_content_df_base.copy()
    questions_with_features_df = extract_selected_features(current_question_content_df, numerical_cols_for_run, categorical_cols_for_run)

    if FORMATTED_TEXT_COL not in questions_with_features_df.columns and QUESTION_TEXT_COL in questions_with_features_df.columns:
         if CORRECT_OPTION_COL not in questions_with_features_df.columns: questions_with_features_df[CORRECT_OPTION_COL] = "A"
         questions_with_features_df[CORRECT_OPTION_COL] = questions_with_features_df[CORRECT_OPTION_COL].fillna('A').astype(str)
         questions_with_features_df[FORMATTED_TEXT_COL] = questions_with_features_df.apply(
             lambda row: format_question_text(row, title_col=QUESTION_TEXT_COL, option_cols=OPTION_COLS, correct_option_col=CORRECT_OPTION_COL), axis=1)
    elif FORMATTED_TEXT_COL not in questions_with_features_df.columns:
        logger.warning(f"Cannot create {FORMATTED_TEXT_COL} as {QUESTION_TEXT_COL} is missing.")
        questions_with_features_df[FORMATTED_TEXT_COL] = ""

    logger.info(f"Generating embeddings for {exp_config_name}...")
    question_embeddings = {
        "question_ids": questions_with_features_df[QUESTION_ID_COL].tolist(),
        "formatted_embeddings": {},
    }
    if not questions_with_features_df.empty and FORMATTED_TEXT_COL in questions_with_features_df.columns:
        question_embeddings["formatted_embeddings"] = generate_text_embeddings(
            data_df=questions_with_features_df, text_col=FORMATTED_TEXT_COL, id_col=QUESTION_ID_COL,
            model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE,
        )
    embedding_dimension = next(iter(question_embeddings["formatted_embeddings"].values())).shape[0] if question_embeddings["formatted_embeddings"] else 0
    if embedding_dimension == 0: logger.warning(f"Embedding dimension is 0 for {exp_config_name}.")

    logger.info(f"Merging datasets for splitting ({exp_config_name})...")
    complete_dataset_df = student_answers_df.merge(
        questions_with_features_df, on=QUESTION_ID_COL, how='inner'
    ).merge(
        original_1pl_irt_params_df[[QUESTION_ID_COL, IRT_DIFFICULTY_COL]], on=QUESTION_ID_COL, how='inner'
    )
    if complete_dataset_df.empty: 
        logger.error(f"Complete dataset for {exp_config_name} is empty. Halting this config."); return None, None, None
    complete_dataset_df[IRT_DIFFICULTY_COL].fillna(complete_dataset_df[IRT_DIFFICULTY_COL].median(), inplace=True)

    logger.info(f"Splitting data ({exp_config_name})...")
    train_q_ids, val_q_ids, test_q_ids = stratified_question_split_3way(
        df=complete_dataset_df, question_col=QUESTION_ID_COL,
        stratify_cols=[CORRECTNESS_COL, IRT_DIFFICULTY_COL],
        test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED, n_bins=5
    )
    if not train_q_ids or not val_q_ids or not test_q_ids: 
        logger.error(f"Data split resulted in empty q_id lists for {exp_config_name}. Halting this config."); return None, None, None

    logger.info(f"Preparing NN datasets for {exp_config_name}...")
    train_ds, train_y, val_ds, val_y, preprocessors, final_actual_num_cols = prepare_nn_datasets(
        merged_df=complete_dataset_df, combined_embeddings=question_embeddings,
        train_q_ids=train_q_ids, val_q_ids=val_q_ids,
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL,
        numerical_feature_cols=numerical_cols_for_run, 
        categorical_feature_cols_to_encode=categorical_cols_for_run, 
        embedding_dim=embedding_dimension
    )
    
    nn_hyperparams = { 
        "user_embedding_dim": 8, "dropout_rate": 0.25, "l2_reg": 0.0005,
        "learning_rate": 2e-4, "dense_layers": [64, 32], "epochs": 60, 
        "batch_size": 1024, "patience_es": 12
    }
    logger.info(f"Creating and training NN for {exp_config_name} with hyperparams: {nn_hyperparams}")
    model = create_nn_model(
        user_vocab_size=preprocessors["user_vocab_size"],
        numerical_feature_size=len(final_actual_num_cols),
        embedding_dim=embedding_dimension,
        user_embedding_dim=nn_hyperparams["user_embedding_dim"],
        dropout_rate=nn_hyperparams["dropout_rate"],
        l2_reg=nn_hyperparams["l2_reg"],
        dense_layers_config=nn_hyperparams["dense_layers"]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=nn_hyperparams["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    train_nn_model(
        model, train_ds, train_y, val_ds, val_y,
        epochs=nn_hyperparams["epochs"],
        batch_size=nn_hyperparams["batch_size"],
        patience_es=nn_hyperparams["patience_es"]
    )
    save_nn_model(model, CURRENT_RUN_DIR, filename=f"model_{exp_config_name.replace(' ', '_').lower()}.keras")

    logger.info(f"Evaluating {exp_config_name} on hold-out test data...")
    test_dataset_inputs, test_labels = make_dataset(
        df=complete_dataset_df, q_ids=test_q_ids, preprocessors=preprocessors,
        combined_embeddings=question_embeddings, user_col=USER_ID_COL, 
        question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL
    )
    test_metrics = evaluate_model(model, test_dataset_inputs, test_labels)
    logger.info(f"Test set correctness performance for {exp_config_name}: {test_metrics}")

    # --- ADDED: Prediction-based IRT and Comparison ---
    logger.info(f"Generating prediction probability matrix for {exp_config_name}...")
    prob_matrix = prediction_matrix(
        data_df=complete_dataset_df, # Use the complete_dataset_df for this config
        q_ids=test_q_ids, # Predictions for test set questions
        preprocessors=preprocessors,
        model=model,
        combined_embeddings=question_embeddings,
        user_col=USER_ID_COL, question_col=QUESTION_ID_COL
    )
    prob_matrix_filename = f"04_pred_probs_{exp_config_name.replace(' ', '_').lower()}.csv"
    prob_matrix.to_csv(os.path.join(CURRENT_RUN_DIR, prob_matrix_filename))
    logger.info(f"Prediction matrix saved for {exp_config_name}.")

    predicted_irt_params_df = local_difficulty_from_binarized_predictions_1pl(prob_matrix)
    predicted_irt_filename = f"05_predicted_1pl_params_{exp_config_name.replace(' ', '_').lower()}.csv"
    predicted_irt_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, predicted_irt_filename), index=False)
    logger.info(f"Predicted 1PL IRT params saved for {exp_config_name}.")
    
    comparison_metrics = {}
    if not original_1pl_irt_params_df.empty and not predicted_irt_params_df.empty:
        comparison_metrics = compare_difficulty(original_1pl_irt_params_df, predicted_irt_params_df, model_type="1pl")
        logger.info(f"Difficulty comparison metrics for {exp_config_name} (1PLv1PL): {comparison_metrics}")
    else:
        logger.warning(f"Skipping difficulty comparison for {exp_config_name} due to empty IRT params DFs.")

    save_preprocessors(preprocessors, CURRENT_RUN_DIR, filename=f"preprocessors_{exp_config_name.replace(' ', '_').lower()}.pkl")

    return test_metrics, final_actual_num_cols, comparison_metrics # Return comparison_metrics

def main_feature_ablation_study():
    student_answers_df, question_content_df_base = load_and_preprocess_data_no_filter()
    
    # Calculate original 1PL IRT difficulties once for stratification purposes
    logger.info("Calculating original 1PL IRT for stratification...")
    original_1pl_irt_params_df = estimate_irt_1pl_difficulty(
        response_df=student_answers_df,
        user_col=USER_ID_COL,
        question_col=QUESTION_ID_COL,
        correctness_col=CORRECTNESS_COL,
    )
    if original_1pl_irt_params_df.empty:
        logger.error("Original 1PL IRT for stratification failed. Halting.")
        return

    # Save the benchmark IRT params for this run
    original_1pl_irt_params_df.to_csv(os.path.join(CURRENT_RUN_DIR, "00_benchmark_original_1pl_irt_params.csv"), index=False)
    logger.info(f"Benchmark original 1PL IRT parameters saved to {os.path.join(CURRENT_RUN_DIR, '00_benchmark_original_1pl_irt_params.csv')}")

    # Define feature sets for ablation based on the new strategy
    ablation_configs = {
        "Model_1_Embeddings_Only": {
            "numerical_cols": [],
            "categorical_cols": []
        },
        "Model_2_Embeddings_QuestionFeatures": {
            "numerical_cols": list(set(QUESTION_FEATURES_NUMERICAL)),
            "categorical_cols": list(set(QUESTION_FEATURES_CATEGORICAL))
        },
        "Model_3_Embeddings_Question_OptionFeatures": {
            "numerical_cols": list(set(QUESTION_FEATURES_NUMERICAL + OPTION_FEATURES_NUMERICAL)),
            "categorical_cols": list(set(QUESTION_FEATURES_CATEGORICAL + OPTION_FEATURES_CATEGORICAL))
        },
        "Model_4_Embeddings_Question_Option_LLMFeatures": {
            "numerical_cols": list(set(QUESTION_FEATURES_NUMERICAL + OPTION_FEATURES_NUMERICAL + LLM_DERIVED_NUMERICAL)),
            "categorical_cols": list(set(QUESTION_FEATURES_CATEGORICAL + OPTION_FEATURES_CATEGORICAL + LLM_DERIVED_CATEGORICAL))
        }
    }

    all_ablation_results = []

    for exp_config_name, features_config in ablation_configs.items():
        test_metrics, final_actual_num_cols, comparison_metrics = run_ablation_experiment(
            exp_config_name, 
            list(set(features_config["numerical_cols"])),
            list(set(features_config["categorical_cols"])),
            student_answers_df.copy(), 
            question_content_df_base.copy(),
            original_1pl_irt_params_df
        )
        if test_metrics:
            all_ablation_results.append({
                "config_name": exp_config_name,
                "numerical_features_config": sorted(list(set(features_config["numerical_cols"]))),
                "categorical_features_config": sorted(list(set(features_config["categorical_cols"]))),
                "final_model_input_feature_count": len(final_actual_num_cols) if final_actual_num_cols else None,
                "test_metrics_correctness": test_metrics,
                "irt_comparison_metrics": comparison_metrics
            })
    
    dump_json(all_ablation_results, os.path.join(CURRENT_RUN_DIR, "feature_ablation_new_buckets_results.json"))
    logger.info(f"\nFeature ablation study (new buckets strategy) complete. Results saved to: {os.path.join(CURRENT_RUN_DIR, 'feature_ablation_new_buckets_results.json')}")
    for res in all_ablation_results:
        auc_score = res['test_metrics_correctness'].get('auc', 'N/A')
        accuracy_score = res['test_metrics_correctness'].get('accuracy', 'N/A')
        pearson_corr = res['irt_comparison_metrics'].get('pearson', 'N/A')
        spearman_corr = res['irt_comparison_metrics'].get('spearman', 'N/A')
        
        auc_str = f"{auc_score:.4f}" if isinstance(auc_score, float) else auc_score
        acc_str = f"{accuracy_score:.4f}" if isinstance(accuracy_score, float) else accuracy_score
        pearson_str = f"{pearson_corr:.4f}" if isinstance(pearson_corr, float) else pearson_corr
        spearman_str = f"{spearman_corr:.4f}" if isinstance(spearman_corr, float) else spearman_corr
        
        logger.info(f"Config: {res['config_name']}, AUC: {auc_str}, Accuracy: {acc_str}, Pearson: {pearson_str}, Spearman: {spearman_str}")

if __name__ == "__main__":
    main_feature_ablation_study() 
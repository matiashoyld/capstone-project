# Comprehensive Modeling Strategy for Question Difficulty Prediction

This document outlines the end-to-end strategy employed for predicting question difficulty, focusing on the final approach that yielded the best correlation with 1-Parameter Logistic (1PL) Item Response Theory (IRT) difficulty scores. This strategy is primarily embodied in the `code/run_best_model_no_filter.py` script and leverages various custom Python modules.

## 1. Overall Goal

The primary objective is to develop a system that can predict the difficulty of educational questions. The approach involves training a neural network (NN) to predict student correctness on questions. The outputs of this NN are then used to estimate IRT difficulty parameters, which are subsequently compared against originally calculated IRT difficulty parameters to evaluate the model's ability to capture inherent question difficulty.

The target correlation metrics are Pearson and Spearman correlations. Additionally, an RMSE-based efficiency evaluation is performed to understand how the NN-derived difficulty compares to traditional IRT in terms of data requirements for achieving a similar level of absolute accuracy (after scale alignment).

## 2. Data Sources

The pipeline utilizes three main data files located in `data/zapien/`:

1.  **`answers.csv`**: Contains student response data.
    *   Initial records: 269,171 rows, 16 columns.
    *   Key columns used: `user_id`, `question_id`, `is_correct`.
2.  **`questions.csv`**: Contains the primary content and metadata for each question.
    *   Initial records: 4,696 rows, 51 columns.
    *   Key columns used: `question_id`, `question_title`, `option_a` - `option_e`, `correct_option_letter`.
3.  **`questions_additional_features.csv`**: Provides supplementary pre-calculated features for questions.
    *   Initial records: 4,483 rows, 19 columns (including `question_id`).

## 3. Baseline IRT Difficulty Calculation (Original Benchmark)

To establish a ground truth for question difficulty, an initial IRT analysis is performed on the **unfiltered** student response data.

*   **Script Reference**: `code/run_best_model_no_filter.py` (function `calculate_original_1pl_irt`).
*   **Module Used**: `modules/irt.py` (function `estimate_irt_1pl_difficulty`).
*   **Model**: 1-Parameter Logistic (1PL) model (Rasch model).
*   **Input Data**: `student_answers_df` (269,171 responses).
    *   Number of unique users for IRT: 1,875.
    *   Number of unique questions for IRT: 9,769.
*   **Process**: 
    *   The `estimate_irt_1pl_difficulty` function uses TensorFlow.
    *   It estimates a difficulty parameter for each question and an ability parameter for each user.
    *   Training runs for up to 100 epochs (Adam optimizer, LR=0.05), with early stopping (patience 10). The example run stopped at epoch 92 (final loss: 0.4365).
    *   Estimated question difficulties are centered by subtracting their mean.
*   **Output**:
    *   A DataFrame `original_1pl_irt_params_df` (9769 questions) is saved as `01_irt_1pl_params_original_no_filter.csv`.
    *   **Summary Statistics for Original 1PL Difficulty (Unfiltered Full Dataset)**:
        *   Count: 9,769
        *   Mean: ~0 (e.g., -8.59e-08)
        *   Std: 1.9015
        *   Min: -5.9035
        *   Max: 6.4527
    *   This `original_1pl_irt_params_df` serves as the primary benchmark for correlation comparisons.

## 4. Feature Engineering and Preprocessing

This stage prepares the question data for the neural network, using the unfiltered approach.

*   **Script Reference**: `code/run_best_model_no_filter.py` (functions `load_and_preprocess_data_no_filter`, `extract_all_features`).
*   **Modules Used**: `modules/features.py`, `modules/utils.py`, `modules.modeling_data.py`.

### 4.1. Initial Data Loading and Merging (`load_and_preprocess_data_no_filter`):
    1.  Data loaded from `answers.csv`, `questions.csv`, and `questions_additional_features.csv`.
    2.  `additional_features_df` is merged (left join) with `question_content_df`. Resulting shape: (4696, 69).

### 4.2. NaN Handling for Merged Question Features:
    *   **Numerical Features**: Coerced to numeric (errors to NaN). NaNs filled with column medians (e.g., `Answer_Length_Variance` filled 213 NaNs).
    *   **Binary Features**: Coerced to numeric (errors to NaN). NaNs filled with `0` (e.g., `Has_Abstract_Symbols` filled 213 NaNs).
    *   **Categorical Features for OHE**: Converted to string. NaNs filled with `'Unknown'` (e.g., `Knowledge_Dimension` filled 213 NaNs).
    *   **Question Filtering**: Explicitly **skipped** for `student_answers_df`.
    *   `question_content_df` is filtered to questions present in `student_answers_df`. Shape: (4696, 69).

### 4.3. Text-Based and Option-Based Feature Extraction (`extract_all_features`):
    *   Text and option features are extracted using functions from `modules/features.py`.
    *   All features in `BASE_NUMERICAL_FEATURE_COLS` are ensured to be numeric, with NaNs filled with 0.
    *   Resulting `questions_with_features_df` shape: (4696, 79).

### 4.4. Text Formatting for Embeddings (`format_question_text` in `modules/utils.py`):
    *   Question title and options combined into `FORMATTED_TEXT_COL`.

### 4.5. Final Merging for NN (`main_best_model_pipeline`):
    *   `student_answers_df` (unfiltered), `questions_with_features_df`, and `original_1pl_irt_params_df` are merged (inner joins).
    *   Resulting `complete_dataset_df` shape (example run): (251851, 96).
    *   NaNs in `IRT_DIFFICULTY_COL` filled with median.

### 4.6. Data Splitting (`stratified_question_split_3way` in `modules/modeling_data.py`):
    *   `complete_dataset_df` is split by `question_id`.
    *   Stratification on `CORRECTNESS_COL` and original 1PL `IRT_DIFFICULTY_COL` (`n_bins=5`).
    *   Split sizes (example run): Train questions: 3286, Validation: 940, Test: 470.

### 4.7. NN Dataset Preparation (`prepare_nn_datasets` in `modules/modeling_data.py`):
    *   Includes numeric coercion, One-Hot Encoding for `CATEGORICAL_FEATURE_COLS_TO_ENCODE`, scaling (StandardScaler), User ID encoding, and embedding lookup.
    *   Number of final numerical features for the model (`final_num_cols` count): 48.

## 5. Text Embedding Generation

*   **Module**: `modules/embeddings.py` (`generate_text_embeddings`).
*   **Model**: `nomic-ai/modernbert-embed-base`.
*   **Process**: 4696 formatted texts embedded in batches. Embeddings are L2 normalized.
*   **Output Embedding Dimension**: 768.

## 6. Neural Network (NN) Architecture

*   **Module**: `modules/neural_net.py` (`create_nn_model`).
*   **Best Configuration (`config_F_lower_lr_more_patience` as used in `run_best_model_no_filter.py`):
    *   `user_embedding_dim`: 8
    *   `dropout_rate`: 0.25
    *   `l2_reg`: 0.0005
    *   `learning_rate`: 0.0002 (2e-4)
    *   `dense_layers_config`: `[64, 32]`.
*   **Inputs**: User ID, Numerical Features (48), Text Embeddings (dim 768).
*   **Layers**: See section 6 of previous version of this report for detailed pathway breakdown (User, Numerical, Embedding pathways, Concatenation, Shared Dense Layers with specified units, ReLU, L2, Dropout, Sigmoid output).
*   **Compilation**: Optimizer `Adam` (LR=0.0002), Loss `binary_crossentropy`, Metrics: `accuracy`, `AUC`.

## 7. Neural Network Training

*   **Module**: `modules/neural_net.py` (`train_nn_model`).
*   **Configuration (`config_F_lower_lr_more_patience`):
    *   `epochs`: 60 (max).
    *   `batch_size`: 1024.
    *   `patience_es` (EarlyStopping): 12.
*   **Process**: Early stopping monitors `val_loss` (mode `min`), `restore_best_weights=True`.
    *   Example run (`run_best_model_no_filter.py`): Restored best weights from epoch 59. Best `val_loss`: 0.5169, `val_accuracy` at best val_loss epoch: 0.7460, `val_auc` at best val_loss epoch: 0.7733.

## 8. Prediction-Based IRT for Difficulty Estimation

*   **Modules**: `modules/evaluation.py::prediction_matrix`, local `local_difficulty_from_binarized_predictions_1pl` in `run_best_model_no_filter.py` (calling `modules/irt.py::estimate_irt_1pl_difficulty`).
1.  **Prediction Matrix**: NN predicts correctness probability for user-question pairs in the test set.
2.  **Estimate 1PL IRT**: Probabilities are binarized (0.5 threshold). These are used with `estimate_irt_1pl_difficulty` to get new 1PL difficulties.

## 9. Evaluation and Comparison

### 9.1. NN Classification Performance (Test Set - `run_best_model_no_filter.py`):
*   Accuracy: 0.7529
*   AUC: 0.7789
*   Precision: 0.8010
*   Recall: 0.8719
*   F1-score: 0.8350

### 9.2. Difficulty Correlation (Original 1PL vs. Prediction-Based 1PL - Unfiltered Data):
*   **Pearson: 0.7606**
*   **Spearman: 0.7346**
*   N (common questions in test set): 470

### 9.3. RMSE-Based Efficiency Evaluation (`run_rmse_evaluation.py` based on `run_best_model_no_filter.py` outputs):
    *   This evaluation assesses the absolute accuracy of the NN-derived 1PL difficulty parameters compared to a ground truth, and estimates the equivalent amount of real student data needed to achieve similar accuracy via traditional IRT.
    *   **Ground Truth 1PL Difficulty (Holdout Set):** Calculated using `estimate_irt_1pl_difficulty` on all 25,440 answers for the 470 holdout questions (from the unfiltered dataset). 
        *   Mean: ~0
        *   Standard Deviation: ~1.6775
    *   **NN-Derived 1PL Model RMSE (after scale alignment): 1.1857**.
        *   The difficulty scores from the NN (via binarized predictions and then 1PL IRT estimation) were aligned to the scale (mean and standard deviation) of the ground truth 1PL difficulties before RMSE calculation. This ensures a fair comparison of the pattern of difficulties, independent of arbitrary IRT scaling factors.
    *   **Efficiency Conclusion**: The NN-derived 1PL difficulty estimates achieve an accuracy (RMSE) comparable to using approximately **5,818 real student answers** in a traditional 1PL IRT estimation for these same 470 holdout questions. This represents approximately **22.87%** of the total available real responses for these items in the holdout set.
    *   **RMSE for Real Data Samples (1PL IRT, from simulation with 0.05 increments):
        *   1.0% (254 answers): RMSE ~1.6807
        *   5.0% (1,272 answers): RMSE ~1.6059
        *   10.0% (2,544 answers): RMSE ~1.4834
        *   15.0% (3,816 answers): RMSE ~1.3666
        *   20.0% (5,088 answers): RMSE ~1.2475
        *   25.0% (6,360 answers): RMSE ~1.1399  *(The NN-derived RMSE of 1.1857 falls between the 20% and 25% real data marks)*
        *   30.0% (7,632 answers): RMSE ~1.0680
        *   35.0% (8,904 answers): RMSE ~1.0025
        *   40.0% (10,176 answers): RMSE ~0.9188
        *   45.0% (11,448 answers): RMSE ~0.8447
        *   50.0% (12,720 answers): RMSE ~0.7815
        *   55.0% (13,992 answers): RMSE ~0.7191
        *   60.0% (15,264 answers): RMSE ~0.6631
        *   65.0% (16,536 answers): RMSE ~0.6088
        *   70.0% (17,808 answers): RMSE ~0.5425
        *   75.0% (19,080 answers): RMSE ~0.4766
        *   80.0% (20,352 answers): RMSE ~0.3960
        *   85.0% (21,624 answers): RMSE ~0.3390
        *   90.0% (22,896 answers): RMSE ~0.2647
        *   95.0% (24,168 answers): RMSE ~0.1821
        *   100.0% (25,440 answers): RMSE ~0.0185

## 10. Key Decisions & Rationale in Final Strategy

*   **No Question Filtering**: Most impactful decision. Including all questions provided a wider original IRT difficulty spread (Std Dev ~1.90 for full dataset, ~1.68 for holdout set), leading to substantially higher correlations (Pearson ~0.76) with NN-derived difficulties.
*   **1PL vs. 1PL Comparison**: This became the primary focus as it yielded the best and most stable correlations.
*   **Binarized Predictions for Prediction-Based IRT**: Retained due to better performance and interpretability with current IRT functions compared to using raw probabilities.
*   **Chosen NN Hyperparameters**: `config_F_lower_lr_more_patience` (detailed above) was optimal for the 1PL unfiltered data scenario.
*   **Comprehensive Features & Robust Preprocessing**: Essential for model performance.

This detailed strategy, centered around using unfiltered data with a 1PL IRT framework and a carefully tuned neural network, represents the current optimal approach for predicting question difficulty, achieving both strong correlation and quantifiable data efficiency. 
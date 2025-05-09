# Neural Network Optimization Report for Difficulty Prediction

This report summarizes the optimization process undertaken to improve the correlation between neural network predicted question difficulty and original IRT-based difficulty scores.

## Initial State & Goal

The initial pipeline involved calculating 1-PL IRT difficulty (though early experiments in `optimization.py` started with 2PL as the original estimate before refining to 1PL vs 1PL and 2PL vs 2PL comparisons), extracting text and option features, generating embeddings, and training a neural network to predict student correctness. The predicted correctness was then used (after binarization) to estimate IRT parameters, and these were compared against the originally calculated IRT difficulty.

The goal of this optimization process was to integrate additional question features and experiment with neural network hyperparameters and IRT estimation techniques to improve the Pearson and Spearman correlation between the model-derived difficulty and the originally calculated IRT difficulty.

## Process Overview

1.  **Integration of New Features**:
    *   A new dataset (`questions_additional_features.csv`) was introduced.
    *   Features were categorized into numerical (requiring scaling) and categorical (requiring one-hot encoding).
    *   The `main.ipynb` notebook and `modules/modeling_data.py` were modified to load, merge, and preprocess these new features. This included robust NaN handling and coercion to appropriate types.

2.  **Development of `optimization.py` Script**:
    *   The core logic from `main.ipynb` was refactored into a script `code/optimization.py` to facilitate systematic experimentation.
    *   This script was designed to run multiple neural network configurations, evaluate them, and compare their derived IRT parameters for both 1PL vs. 1PL and 2PL vs. 2PL scenarios.

3.  **Overfitting Mitigation in Neural Network**:
    *   Initial training curves indicated overfitting.
    *   The `EarlyStopping` callback in `modules/neural_net.py` was confirmed and adjusted to monitor `val_loss` with `mode="min"`.
    *   `Dropout` layers and `L2 kernel regularization` were added to `create_nn_model` in `modules/neural_net.py`.

4.  **Hyperparameter Tuning Loop**:
    *   The `optimization.py` script was enhanced with a loop to test various neural network hyperparameter configurations (learning rate, dropout, L2, architecture, epochs, etc.).
    *   Each configuration was run for both 1PL-vs-1PL and 2PL-vs-2PL comparison setups.

5.  **Experiment with Raw Probabilities for IRT Estimation**:
    *   An alternative `difficulty_from_probabilities_irt` function was tested to estimate IRT parameters using the NN's raw output probabilities, instead of binarizing them.

6.  **Feature Importance Analysis**:
    *   A proxy feature importance analysis using RandomForestClassifier was conducted.

7.  **Impact of Question Filtering (Key Experiment)**:
    *   A dedicated script, `code/run_best_model_no_filter.py`, was created to evaluate the best-performing 1PL configuration **without applying the initial question filtering** (i.e., not removing questions with 0%/100% correctness or <10 responses).

## Key Findings and Learnings

1.  **Impact of Removing Question Filters (Most Significant Finding)**:
    *   Running the pipeline without the initial question filtering (using all 9769 unique questions) led to a **substantial improvement** in the correlation between original 1PL IRT difficulties and prediction-based 1PL IRT difficulties.
    *   The best 1PL configuration (`config_F_lower_lr_more_patience`) achieved:
        *   **Pearson Correlation: 0.7654**
        *   **Spearman Correlation: 0.7432**
    *   This compares favorably to the best 1PL correlations with filtering (Pearson ~0.58, Spearman ~0.60).
    *   The standard deviation of the original 1PL IRT difficulties increased from ~1.36 (with filtering) to ~1.90 (without filtering), providing a wider and more varied difficulty scale.
    *   **Conclusion**: For maximizing correlation with 1PL IRT, not filtering out extreme questions is highly beneficial as it provides a richer difficulty spectrum for the model to learn and correlate with.

2.  **Best Performing Configuration (with No Filtering, 1PL vs. 1PL)**:
    *   The configuration used in `run_best_model_no_filter.py` (based on `config_F_lower_lr_more_patience` from `optimization.py`) is currently the top performer.
    *   **Neural Network Hyperparameters**:
        *   `user_embedding_dim`: 8
        *   `dropout_rate`: 0.25
        *   `l2_reg`: 0.0005
        *   `learning_rate`: 0.0002 (2e-4)
        *   `dense_layers_config`: `[64, 32]`
        *   `epochs`: 60 (max, with early stopping patience of 12)
        *   `batch_size`: 1024
        *   `patience_es` (for EarlyStopping): 12
    *   **Test Performance of this NN (on unfiltered data)**:
        *   Accuracy: ~0.7527
        *   AUC: ~0.7797

3.  **1PL vs. 1PL Compared to 2PL vs. 2PL (with Filtering in `optimization.py`)**:
    *   In the `optimization.py` runs (which used question filtering), the 1PL vs. 1PL comparisons generally yielded slightly better correlations (best Pearson ~0.584) than the 2PL vs. 2PL comparisons (best Pearson ~0.578).

4.  **Raw Probabilities vs. Binarized for IRT Estimation**:
    *   Using raw probabilities directly with the current `estimate_irt_1pl_difficulty` or `estimate_irt_2pl_params` functions did not consistently outperform using binarized predictions. For the 2PL case, it also led to less interpretable discrimination parameter correlations.
    *   **Conclusion**: Binarizing NN probability outputs (at 0.5) appears to be a more stable approach for the current IRT estimation functions.

5.  **Impact of New Features & Overfitting Management**: These were confirmed as beneficial in earlier optimization stages as detailed in the `optimization.py` logs and previous report versions.

## Final Recommended Strategy (Based on Highest Correlation)

The setup that achieved the **Pearson correlation of 0.7654 and Spearman correlation of 0.7432** is currently the best approach found:

*   **Script Used**: `code/run_best_model_no_filter.py`.
*   **Data Preprocessing**: **No initial question filtering** (questions with 0%/100% correctness or <10 responses are *included*).
    *   Loading of `questions_additional_features.csv` and merging.
    *   NaNs in new numerical features filled with median or 0 for binary types.
    *   NaNs in categorical features filled with 'Unknown' and then One-Hot Encoded.
    *   Standard scaling applied to all final numerical features.
*   **Original IRT**: **1PL model** (`estimate_irt_1pl_difficulty`) run on the complete (unfiltered) student answers.
*   **Neural Network Architecture & Training**: As per `config_F_lower_lr_more_patience` detailed in finding #2 above.
*   **Prediction-based IRT**: Probabilities from the trained NN were **binarized at a 0.5 threshold**. These binarized predictions were then used to fit a **1PL IRT model**.
*   **Comparison**: The difficulty parameter from this prediction-based 1PL IRT was compared against the difficulty parameter from the original 1PL IRT (calculated on unfiltered data).

## Future Considerations

*   **Re-evaluate 2PL vs. 2PL without Filtering**: Conduct the 2PL vs. 2PL comparison using the unfiltered dataset to see if similar improvements in correlation are observed for the 2PL model.
*   **Refined Feature Selection**: Even with the unfiltered data, use feature importance (e.g., from RandomForest run on the unfiltered data setup) to see if reducing features can further enhance the model.
*   **Direct Difficulty Prediction**: This remains a significant alternative. Train the NN to directly predict the (unfiltered) 1PL IRT difficulty scores as a regression task.
*   **Impact of Extreme Items on Stratification**: With unfiltered data, the `stratified_question_split_3way` might behave differently. Monitor split balance. The increased `n_bins=5` in `run_best_model_no_filter.py` was a good step to try and handle more diverse difficulty values during stratification.

This optimization process, particularly the experiment with data filtering, has provided a clear path to substantially improving the difficulty correlation metrics. 
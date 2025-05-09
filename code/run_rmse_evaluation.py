import os
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import logging
from tqdm import tqdm # For progress bar in script execution

# --- Add Modules to Path (similar to notebook) ---
ROOT_DIR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming this script is in code/
MODULES_PATH = os.path.join(ROOT_DIR_SCRIPT, 'code', 'modules')
if MODULES_PATH not in sys.path:
    sys.path.insert(0, MODULES_PATH)

# Import local modules
from modules.irt import estimate_irt_1pl_difficulty # Changed from 2PL
from modules.evaluation import setup_paths, calculate_rmse # dump_json is not used here directly but good for consistency

# --- Configure Logging and TensorFlow ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
plt.style.use('seaborn-v0_8-whitegrid')

# --- Constants and Parameters (from notebook) ---
USER_ID_COL = "user_id"
QUESTION_ID_COL = "question_id"
CORRECTNESS_COL = "is_correct"
IRT_DIFFICULTY_COL = "difficulty"

NUM_REPETITIONS = 10
SAMPLE_SIZES_PCT = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0
]
MIN_IRT_SAMPLE_SIZE = 200

# --- Main Function to Run the Evaluation Pipeline ---
def run_rmse_efficiency_evaluation():
    # --- 1. Setup Paths ---
    # IMPORTANT: User needs to set this to the output directory of run_best_model_no_filter.py
    # Example: RUN_DIRECTORY = "../results_best_model_no_filter/20250509_080218_1PL_no_filter"
    RUN_DIRECTORY = "../results_best_model_no_filter/20250509_150234_1PL_no_filter" # Placeholder, replace with your actual latest run
    logger.info(f"🚀 Starting RMSE Efficiency Evaluation for run: {RUN_DIRECTORY}")
    
    # Use the best 1PL config name that was used in run_best_model_no_filter.py for consistency
    best_nn_config_name = "best_1pl_config_F_no_filter"
    paths = setup_paths(RUN_DIRECTORY, irt_model_type="1pl", nn_config_name=best_nn_config_name)

    # --- 2. Load Data ---
    logger.info("🔄 Loading data...")
    try:
        answers_df = pd.read_csv(paths['answers_file'])
        logger.info(f"Loaded {len(answers_df):,} student answers from {answers_df[USER_ID_COL].nunique():,} users")
        
        holdout_ids_df = pd.read_csv(paths['holdout_ids_path'])
        holdout_ids = holdout_ids_df[QUESTION_ID_COL].tolist()
        logger.info(f"Loaded {len(holdout_ids):,} holdout question IDs")
        
        # This is the NN-derived 1PL difficulty from the best model run (no filter)
        nn_predicted_difficulty_df = pd.read_csv(paths['predicted_irt_file'])
        logger.info(f"Loaded NN-derived 1PL difficulty estimates for {len(nn_predicted_difficulty_df):,} questions from {paths['predicted_irt_file']}")

    except FileNotFoundError as e:
        logger.error(f"Error loading data files: {e}. Please ensure RUN_DIRECTORY is correct and contains the output from run_best_model_no_filter.py.")
        return

    # --- 3. Filter Data to Holdout Questions ---
    holdout_ids_str = [str(qid) for qid in holdout_ids]
    answers_df[QUESTION_ID_COL] = answers_df[QUESTION_ID_COL].astype(str)
    holdout_answers_df = answers_df[answers_df[QUESTION_ID_COL].isin(holdout_ids_str)].copy()
    logger.info(f"📊 Holdout set: {len(holdout_ids):,} questions, {len(holdout_answers_df):,} answers from {holdout_answers_df[USER_ID_COL].nunique():,} unique students")

    if holdout_answers_df.empty or holdout_answers_df[USER_ID_COL].nunique() < 2 or holdout_answers_df[QUESTION_ID_COL].nunique() < 2:
        logger.error("Error: Insufficient data in holdout answers after filtering for IRT estimation.")
        return
    logger.info("✅ Sufficient data available for ground truth IRT estimation.")

    # --- 4. Calculate Ground Truth 1PL Difficulty ---
    logger.info("📊 Calculating ground truth 1PL difficulty using complete holdout answer set...")
    ground_truth_difficulty_df = estimate_irt_1pl_difficulty(
        response_df=holdout_answers_df,
        user_col=USER_ID_COL,
        question_col=QUESTION_ID_COL,
        correctness_col=CORRECTNESS_COL
    )
    logger.info("✅ Ground truth 1PL difficulty calculated.")
    logger.info(f"Ground Truth 1PL Difficulty Statistics:\n{ground_truth_difficulty_df[IRT_DIFFICULTY_COL].describe()}")

    # --- 5. Calculate RMSE for NN-Derived Predictions ---
    logger.info("🤖 Calculating RMSE for NN-derived 1PL predictions...")
    nn_derived_rmse = calculate_rmse(
        ground_truth_difficulty_df, 
        nn_predicted_difficulty_df, 
        QUESTION_ID_COL,
        IRT_DIFFICULTY_COL
    )
    if pd.isna(nn_derived_rmse):
        logger.warning("Warning: Could not calculate RMSE for NN-derived model.")
    else:
        logger.info(f"📉 NN-Derived 1PL Model RMSE: {nn_derived_rmse:.4f}")

    # --- 6. Simulate Traditional IRT with Varying Real Data (1PL) ---
    logger.info("📈 Simulating traditional 1PL IRT with varying data sizes (RMSE)...")
    real_data_rmse = {}
    max_holdout_answers = len(holdout_answers_df)
    actual_sample_sizes = sorted(list(set([
        int(pct * max_holdout_answers) for pct in SAMPLE_SIZES_PCT 
        if int(pct * max_holdout_answers) >= MIN_IRT_SAMPLE_SIZE
    ])))
    if max_holdout_answers not in actual_sample_sizes and max_holdout_answers >= MIN_IRT_SAMPLE_SIZE:
        actual_sample_sizes.append(max_holdout_answers)

    if not actual_sample_sizes:
        logger.warning(f"Warning: No sample sizes large enough (>{MIN_IRT_SAMPLE_SIZE}) for simulation.")
    else:
        logger.info(f"Sample sizes to test (num answers): {actual_sample_sizes}")
        for n_answers in tqdm(actual_sample_sizes, desc="Testing sample sizes for 1PL IRT"):
            logger.info(f"\nTesting 1PL with {n_answers:,} answers ({NUM_REPETITIONS} repetitions)...")
            rep_rmse = []
            for rep in range(NUM_REPETITIONS):
                if n_answers >= len(holdout_answers_df):
                    sample_answers_df = holdout_answers_df.copy()
                else:
                    sample_answers_df = holdout_answers_df.sample(n=n_answers, random_state=rep)
                
                if sample_answers_df[USER_ID_COL].nunique() < 2 or sample_answers_df[QUESTION_ID_COL].nunique() < 2:
                    logger.warning(f"  Skipping Rep {rep+1}: Insufficient unique users/questions in sample.")
                    continue
                    
                estimated_params_df = estimate_irt_1pl_difficulty(
                    response_df=sample_answers_df,
                    user_col=USER_ID_COL, question_col=QUESTION_ID_COL, correctness_col=CORRECTNESS_COL
                )
                
                if estimated_params_df is None or estimated_params_df.empty:
                    logger.warning(f"  Skipping Rep {rep+1}: 1PL IRT estimation returned empty results.")
                    continue
                    
                rmse = calculate_rmse(
                    ground_truth_difficulty_df, estimated_params_df, QUESTION_ID_COL, IRT_DIFFICULTY_COL
                )
                if not pd.isna(rmse):
                    rep_rmse.append(rmse)
            
            if rep_rmse:
                avg_rmse = np.mean(rep_rmse)
                real_data_rmse[n_answers] = avg_rmse
                logger.info(f"  → Average RMSE for {n_answers:,} answers (1PL): {avg_rmse:.4f}")
            else:
                logger.warning(f"  → No valid RMSE values calculated for {n_answers:,} answers (1PL).")
                real_data_rmse[n_answers] = np.nan

    # --- 7. Plot Results and Interpret ---
    logger.info("📊 Plotting RMSE efficiency comparison results...")
    valid_rmse_plot = {k: v for k, v in real_data_rmse.items() if not pd.isna(v)}

    if not valid_rmse_plot and pd.isna(nn_derived_rmse):
        logger.error("Error: No valid RMSE values to plot.")
    else:
        plt.figure(figsize=(12, 7))
        x_vals, y_vals, x_plot, y_plot = (np.array([]),)*4
        min_real_y, max_real_y = (nn_derived_rmse*0.9 if not pd.isna(nn_derived_rmse) else 0, 
                                  nn_derived_rmse*1.1 if not pd.isna(nn_derived_rmse) else 1)
        min_x, max_x = (MIN_IRT_SAMPLE_SIZE, max_holdout_answers)

        if valid_rmse_plot:
            x_vals = np.array(list(valid_rmse_plot.keys()))
            y_vals = np.array(list(valid_rmse_plot.values()))
            sort_indices_x = np.argsort(x_vals)
            x_plot = x_vals[sort_indices_x]
            y_plot = y_vals[sort_indices_x]
            plt.plot(x_plot, y_plot, marker='o', linestyle='-', label="Traditional 1PL IRT (Real Data RMSE)", color='#3498db', markersize=6)
            min_real_y, max_real_y = (np.min(y_vals), np.max(y_vals))
            min_x, max_x = (np.min(x_vals), np.max(x_vals))

        intersection_x = np.nan
        intersection_text = ""
        if not pd.isna(nn_derived_rmse):
            plt.axhline(y=nn_derived_rmse, color='#e74c3c', linestyle='--', linewidth=1.5, label=f"NN-Derived 1PL Model RMSE ({nn_derived_rmse:.4f})")
            if len(y_plot) < 2:
                intersection_text = "Not enough real data points for interpolation."
            elif nn_derived_rmse < min_real_y:
                intersection_text = "NN-Derived model outperforms all tested real data samples (RMSE)."
                # intersection_x = min_x # For plotting line at the start
            elif nn_derived_rmse > max_real_y:
                intersection_text = "NN-Derived model underperforms all tested real data samples (RMSE)."
                # intersection_x = max_x # For plotting line at the end
            else:
                intersection_x = np.interp(nn_derived_rmse, y_plot, x_plot) # y_plot must be increasing for np.interp
                # If y_plot (RMSE) is generally decreasing with x_plot (sample size), we need to interp on reversed sorted values
                if not np.all(np.diff(y_plot) >= 0):
                    sort_indices_y_asc = np.argsort(y_plot) # Sort RMSE ascending
                    sorted_y_for_interp = y_plot[sort_indices_y_asc]
                    sorted_x_for_interp = x_plot[sort_indices_y_asc]
                    if not np.all(np.diff(sorted_y_for_interp) >=0): # Check if strictly increasing after sort
                         # If there are plateaus, interpolation might be tricky, find first point >= synthetic_rmse
                        idx_above = np.where(sorted_y_for_interp <= nn_derived_rmse)[0]
                        if len(idx_above) > 0:
                            intersection_x = sorted_x_for_interp[idx_above[0]]
                        else: # synthetic_rmse is lower than all real data points
                            intersection_x = min_x # Or handle as outperform case
                    else:
                         intersection_x = np.interp(nn_derived_rmse, sorted_y_for_interp, sorted_x_for_interp)

                intersection_text = f"Equivalent to ~{intersection_x:,.0f} real student answers (RMSE)"
                plt.axvline(x=intersection_x, color='#2ecc71', linestyle=':', linewidth=1.5, label=f"Equiv. Real Answers: ~{intersection_x:,.0f}")
        else:
            intersection_text = "NN-Derived RMSE could not be calculated."

        plt.xlabel("Number of Real Student Answers Used (Holdout Set)", fontsize=12)
        plt.ylabel("1PL IRT Difficulty Estimation RMSE", fontsize=12)
        plt.title("Efficiency Comparison: NN-Derived vs. Real Data for 1PL IRT Estimation (RMSE)", fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='gray')
        plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.3, color='lightgray')
        ax = plt.gca()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='major', labelsize=10)
        if not pd.isna(intersection_x) and not pd.isna(nn_derived_rmse) and not (nn_derived_rmse < min_real_y or nn_derived_rmse > max_real_y) :
            text_x_pos = intersection_x * 1.1 if intersection_x < (min_x + (max_x-min_x)/2) else intersection_x * 0.5
            plt.annotate(intersection_text, xy=(intersection_x, nn_derived_rmse), 
                         xytext=(text_x_pos, nn_derived_rmse * (1.05 if nn_derived_rmse < (min_real_y + (max_real_y-min_real_y)/2) else 0.95) ), 
                         arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                         fontsize=9, ha='center')
        else:
            plt.text(0.05, 0.90, intersection_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(paths['plot_save_path'], dpi=300)
        logger.info(f"💾 RMSE Plot saved to {paths['plot_save_path']}")
        # plt.show() # Comment out for script execution without display blocking

    # --- 8. Results Summary and Interpretation ---
    logger.info("\n--- 📋 RMSE Efficiency Summary ---")
    if not pd.isna(nn_derived_rmse):
        logger.info(f"NN-Derived 1PL Model RMSE: {nn_derived_rmse:.4f}")
    else:
        logger.info("NN-Derived 1PL Model RMSE: Could not be calculated.")
    logger.info("\nAvg RMSE using Real Data Samples (1PL):")
    if valid_rmse_plot:
        rmse_df = pd.DataFrame({
            'Sample Size': list(real_data_rmse.keys()),
            'Average RMSE': list(real_data_rmse.values()),
            'Sample %': [n/max_holdout_answers*100 if max_holdout_answers > 0 else 0 for n in real_data_rmse.keys()]
        }).sort_values('Sample Size')
        logger.info(f"\n{rmse_df.to_string(formatters={'Sample Size': '{:,.0f}'.format, 'Average RMSE': '{:.4f}'.format, 'Sample %': '{:.1f}%'.format})}")
    else:
        logger.info("No valid real data RMSEs were calculated.")
    logger.info(f"\nEfficiency Conclusion (RMSE based): {intersection_text}")

if __name__ == "__main__":
    run_rmse_efficiency_evaluation() 
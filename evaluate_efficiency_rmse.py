import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import sys

# --- Add modules directory to sys.path ---
# Assumes the script is run from the project root directory (e.g., capstone-new)
# and the modules are in ./code/modules
script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the script itself
# project_root = os.path.abspath(os.path.join(script_dir, '..')) # Assumes script is one level down
project_root = os.getcwd() # Assume script is run from project root
modules_path = os.path.join(project_root, 'code', 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Now import local modules
try:
    from irt import estimate_irt_2pl_params 
    # from evaluation import prediction_matrix, difficulty_from_predictions # Not needed
    # from modeling_data import load_preprocessors # Not needed
    # from neural_net import create_nn_model # Not needed
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Ensure the 'code/modules' directory is correctly added to PYTHONPATH or relative to this script.")
    sys.exit(1)

# Configure logging
log_file_path = 'evaluate_efficiency_rmse.log' # Log in the same directory as the script
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants and Parameters ---
USER_ID_COL = "user_id"
QUESTION_ID_COL = "question_id"
CORRECTNESS_COL = "is_correct"
IRT_DIFFICULTY_COL = "difficulty"

NUM_REPETITIONS = 10
SAMPLE_SIZES_PCT = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# --- Utility Functions ---

def calculate_rmse(true_difficulty_df, estimated_difficulty_df, question_col, irt_col):
    """Calculates RMSE between true and estimated difficulties."""
    # Ensure question_id columns are of the same type for merging
    true_difficulty_df[question_col] = true_difficulty_df[question_col].astype(str)
    estimated_difficulty_df[question_col] = estimated_difficulty_df[question_col].astype(str)

    merged = pd.merge(
        true_difficulty_df[[question_col, irt_col]],
        estimated_difficulty_df[[question_col, irt_col]], # Only need difficulty and ID here
        on=question_col,
        suffixes=("_true", "_est")
    )
    
    if merged.empty or len(merged) < 2:
        logger.warning("Not enough matching questions (<2) for RMSE calculation.")
        return np.nan

    true_col = f"{irt_col}_true"
    est_col = f"{irt_col}_est"
    
    if true_col not in merged.columns or est_col not in merged.columns:
        logger.error(f"Required columns not found after merge: {true_col}, {est_col}")
        return np.nan # Return NaN instead of raising error

    merged = merged[[true_col, est_col]].dropna()
    if len(merged) < 2:
        logger.warning("Not enough non-NaN pairs (<2) for RMSE calculation.")
        return np.nan
    
    # Check for constant series which would lead to division by zero in variance calc
    if merged[true_col].nunique() <= 1 or merged[est_col].nunique() <= 1:
        logger.warning("One or both difficulty series are constant. RMSE calculation might be misleading or error.")
        # Return 0 if they are constant and equal, otherwise could return NaN or large number
        if merged[true_col].nunique() <= 1 and merged[est_col].nunique() <= 1 and merged[true_col].iloc[0] == merged[est_col].iloc[0]:
             return 0.0
        # else: # Or handle differently if needed
        #    return np.nan

    try:
        rmse = np.sqrt(mean_squared_error(merged[true_col], merged[est_col]))
    except ValueError as e:
        logger.error(f"Error calculating RMSE: {e}")
        return np.nan
    return rmse

def setup_paths(run_dir_relative):
    """Set up all file paths needed for the evaluation, assuming script is run from project root."""
    root_dir = os.getcwd() 
    run_dir_abs = os.path.join(root_dir, run_dir_relative)
    
    paths = {
        'data_dir': os.path.join(root_dir, "data", "zapien"),
        'run_dir': run_dir_abs, # Store absolute path
        'plot_save_path': os.path.join(run_dir_abs, "efficiency_evaluation_rmse.png")
    }
    
    paths['answers_file'] = os.path.join(paths['data_dir'], "answers.csv")
    paths['holdout_ids_path'] = os.path.join(paths['run_dir'], "holdout_ids.csv")
    paths['predicted_irt_file'] = os.path.join(paths['run_dir'], "05_predicted_2pl_params.csv")
    
    # Check if essential files exist
    for key, path in paths.items():
        if key not in ['plot_save_path'] and not os.path.exists(path):
             logger.warning(f"Path does not exist: {path} for key {key}")
             if key in ['answers_file', 'holdout_ids_path', 'predicted_irt_file']:
                 raise FileNotFoundError(f"Essential file not found: {path}")

    return paths

# --- Main Execution ---
def main(run_directory_relative):
    try:
        paths = setup_paths(run_directory_relative)
    except FileNotFoundError as e:
        logger.error(f"Setup failed: {e}")
        return
        
    logger.info(f"🚀 Starting RMSE Efficiency Evaluation for run: {run_directory_relative}")

    # --- Load Data ---
    logger.info("🔄 Loading data...")
    try:
        answers_df = pd.read_csv(paths['answers_file'])
        holdout_ids_df = pd.read_csv(paths['holdout_ids_path'])
        holdout_ids = holdout_ids_df[QUESTION_ID_COL].tolist()
        synthetic_difficulty_df = pd.read_csv(paths['predicted_irt_file'])
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        return

    # Filter answers to only holdout questions
    holdout_ids_str = [str(qid) for qid in holdout_ids]
    answers_df[QUESTION_ID_COL] = answers_df[QUESTION_ID_COL].astype(str)
    holdout_answers_df = answers_df[answers_df[QUESTION_ID_COL].isin(holdout_ids_str)].copy()
    logger.info(f"📊 Holdout set: {len(holdout_ids):,} questions, {len(holdout_answers_df):,} answers")
    
    if holdout_answers_df.empty or holdout_answers_df[USER_ID_COL].nunique() < 2 or holdout_answers_df[QUESTION_ID_COL].nunique() < 2:
        logger.error("Insufficient data in holdout answers df after filtering.")
        return

    # --- Calculate Ground Truth (RMSE Context) ---
    logger.info("📊 Calculating RMSE ground truth difficulty using complete holdout answer set (2PL)...")
    try:
        holdout_params_true_df = estimate_irt_2pl_params(
            response_df=holdout_answers_df,
            user_col=USER_ID_COL,
            question_col=QUESTION_ID_COL,
            correctness_col=CORRECTNESS_COL,
            reg_lambda=0.0 
        )
        # Ensure the returned DataFrame is not empty
        if holdout_params_true_df is None or holdout_params_true_df.empty:
             logger.error("Ground truth IRT estimation returned empty or None.")
             return
             
        ground_truth_difficulty_df = holdout_params_true_df[[QUESTION_ID_COL, IRT_DIFFICULTY_COL]].copy()
        logger.info("✅ RMSE Ground truth difficulty calculated.")
        logger.info("Ground Truth Difficulty (2PL) Stats:")
        logger.info(f"\n{ground_truth_difficulty_df[IRT_DIFFICULTY_COL].describe()}")
        # Save ground truth for reference if needed
        # ground_truth_difficulty_df.to_csv(os.path.join(paths['run_dir'], 'ground_truth_holdout_difficulty.csv'), index=False)
    except Exception as e:
        logger.error(f"Error calculating ground truth IRT: {e}", exc_info=True)
        return

    # --- Calculate Synthetic Model RMSE ---
    logger.info("🤖 Calculating RMSE for synthetic predictions...")
    synthetic_rmse = calculate_rmse(
        ground_truth_difficulty_df, 
        synthetic_difficulty_df, 
        QUESTION_ID_COL,
        IRT_DIFFICULTY_COL
    )
    if pd.isna(synthetic_rmse):
        logger.warning("Could not calculate RMSE for synthetic model.")
        # Continue simulation even if synthetic RMSE is NaN, maybe real data works
    else:
        logger.info(f"📉 Synthetic Model RMSE: {synthetic_rmse:.4f}")

    # --- Simulate Traditional IRT with Varying Real Data (RMSE) ---
    logger.info("📈 Simulating traditional IRT with varying data sizes (RMSE)...")
    real_data_rmse = {}
    max_holdout_answers = len(holdout_answers_df)
    min_irt_sample_size = 200 
    actual_sample_sizes = sorted(list(set([int(pct * max_holdout_answers) for pct in SAMPLE_SIZES_PCT if int(pct * max_holdout_answers) >= min_irt_sample_size])))
    if max_holdout_answers not in actual_sample_sizes and max_holdout_answers >= min_irt_sample_size:
        actual_sample_sizes.append(max_holdout_answers)
    elif not actual_sample_sizes and max_holdout_answers >= min_irt_sample_size:
         actual_sample_sizes.append(max_holdout_answers)
    
    if not actual_sample_sizes:
        logger.warning(f"No sample sizes large enough (>{min_irt_sample_size}) for simulation. Skipping simulation loop.")
    else:
        logger.info(f"Sample sizes (num answers): {actual_sample_sizes}")
        for n_answers in actual_sample_sizes:
            logger.info(f"  Testing with {n_answers:,} answers ({NUM_REPETITIONS} reps)...") 
            rep_rmse = []
            for rep in range(NUM_REPETITIONS):
                logger.debug(f"    Rep {rep+1}/{NUM_REPETITIONS}...")
                if n_answers >= len(holdout_answers_df):
                    sample_answers_df = holdout_answers_df.copy() # Use copy to be safe
                else:
                    sample_answers_df = holdout_answers_df.sample(n=n_answers, random_state=rep)
                
                if sample_answers_df[USER_ID_COL].nunique() < 2 or sample_answers_df[QUESTION_ID_COL].nunique() < 2:
                    logger.warning(f"    Skipping Rep {rep+1}: Insufficient unique users/questions in sample.")
                    continue
                    
                try:
                    estimated_params_df = estimate_irt_2pl_params(
                        response_df=sample_answers_df,
                        user_col=USER_ID_COL,
                        question_col=QUESTION_ID_COL,
                        correctness_col=CORRECTNESS_COL,
                        reg_lambda=0.0 
                    )
                    if estimated_params_df is None or estimated_params_df.empty:
                         logger.warning(f"    Skipping Rep {rep+1}: IRT estimation returned empty or None.")
                         continue
                         
                    rmse = calculate_rmse(
                        ground_truth_difficulty_df,
                        estimated_params_df,
                        QUESTION_ID_COL,
                        IRT_DIFFICULTY_COL
                    )
                    if not pd.isna(rmse):
                        rep_rmse.append(rmse)
                        logger.debug(f"      Rep {rep+1} RMSE: {rmse:.4f}")
                    else:
                        logger.warning(f"      Rep {rep+1} RMSE calculation returned NaN.")
                except Exception as e:
                    logger.warning(f"    IRT estimation or RMSE Calc failed for N={n_answers}, Rep {rep+1}: {e}", exc_info=False) # Set exc_info=False for less verbose logs
            
            if rep_rmse:
                avg_rmse = np.mean(rep_rmse)
                real_data_rmse[n_answers] = avg_rmse
                logger.info(f"  -> Avg RMSE for N={n_answers:,}: {avg_rmse:.4f}")
            else:
                logger.info(f"  -> No valid RMSE calculated for N={n_answers:,}.")
                real_data_rmse[n_answers] = np.nan

    # --- Plot Results ---
    logger.info("📊 Plotting RMSE efficiency comparison results...")
    valid_rmse = {k: v for k, v in real_data_rmse.items() if not pd.isna(v)}
    if not valid_rmse and pd.isna(synthetic_rmse):
        logger.error("No valid RMSE values to plot. Exiting.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.figure(figsize=(10, 6))

    if valid_rmse:
        # Convert to numpy arrays for easier sorting
        x_vals = np.array(list(valid_rmse.keys()))
        y_vals = np.array(list(valid_rmse.values()))
        
        # Sort by x_vals (sample size) initially for plotting
        sort_indices_x = np.argsort(x_vals)
        x_plot = x_vals[sort_indices_x]
        y_plot = y_vals[sort_indices_x]

        plt.plot(x_plot, y_plot, marker='o', linestyle='-', label="Traditional IRT (Real Data RMSE)", color='#3498db', markersize=6)
        
        min_real_y = np.min(y_vals)
        max_real_y = np.max(y_vals)
        min_x = np.min(x_vals)
        max_x = np.max(x_vals)
    else:
        # Set default ranges if no real data points
        min_real_y, max_real_y = (synthetic_rmse*0.9, synthetic_rmse*1.1) if not pd.isna(synthetic_rmse) else (0,1)
        min_x, max_x = (100, max_holdout_answers) # Default x range
        x_vals = np.array([]) # Ensure x_vals is defined as numpy array
        y_vals = np.array([])
        x_plot = np.array([])
        y_plot = np.array([])

    if not pd.isna(synthetic_rmse):
        plt.axhline(y=synthetic_rmse, color='#e74c3c', linestyle='--', linewidth=1.5, label=f"Synthetic Model RMSE ({synthetic_rmse:.4f})")

    intersection_x = np.nan
    # --- Determine Intersection Text ---
    if pd.isna(synthetic_rmse):
         intersection_text = "Synthetic RMSE could not be calculated."
    elif len(y_vals) < 2:
         intersection_text = "Not enough real data points for comparison."
    elif synthetic_rmse < min_real_y:
        intersection_text = "Synthetic model outperforms all tested real data samples (RMSE)."
    elif synthetic_rmse > max_real_y:
        intersection_text = "Synthetic model underperforms all tested real data samples (RMSE)."
    else:
        # Interpolation is possible
        # Sort by y_vals (RMSE) ASCENDING for np.interp
        sort_indices_y = np.argsort(y_vals)
        sorted_y_asc = y_vals[sort_indices_y]
        sorted_x_corr = x_vals[sort_indices_y]

        try:
            # np.interp requires x points (sorted_y_asc) to be increasing
            intersection_x = np.interp(synthetic_rmse, sorted_y_asc, sorted_x_corr)
            # Validate interpolation result
            if pd.isna(intersection_x) or intersection_x < min_x or intersection_x > max_x:
                 logger.warning(f"Interpolation resulted in unexpected value: {intersection_x}. Setting to NaN.")
                 intersection_x = np.nan
                 intersection_text = "Intersection could not be accurately determined via interpolation."
            else:
                 intersection_text = f"Equivalent to ~{intersection_x:,.0f} real student answers (RMSE)"
                 plt.axvline(x=intersection_x, color='#2ecc71', linestyle=':', linewidth=1.5, label=f"Equiv. Real Answers: ~{intersection_x:,.0f}")

        except ValueError as e:
            logger.warning(f"Interpolation failed: {e}. Check if RMSE values are unique.")
            intersection_x = np.nan # Set to NaN if interpolation fails
            intersection_text = "Intersection could not be determined due to interpolation error."


    plt.xlabel("Number of Real Student Answers Used (Holdout Set)", fontsize=12)
    plt.ylabel("IRT Difficulty Estimation RMSE", fontsize=12)
    plt.title("Efficiency Comparison (RMSE): Synthetic vs. Real Data for IRT Estimation", fontsize=14, fontweight='bold')
    plt.xscale('log') 
    plt.yscale('linear') 
    plt.legend(fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='gray')
    plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.3, color='lightgray')
    plt.grid(False, which='major', axis='x')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # --- Annotation Logic (Revised) ---
    # Only annotate if interpolation was successful and synthetic RMSE is valid
    if not pd.isna(intersection_x) and not pd.isna(synthetic_rmse): 
        # Adjust text position dynamically
        y_range = max(y_plot.tolist()+[synthetic_rmse]) - min(y_plot.tolist()+[synthetic_rmse]) if len(y_plot)>0 else 0.1
        text_x_offset = 0.1 * (np.log10(max_x) - np.log10(min_x)) if min_x > 0 else 0.1
        text_x = 10**(np.log10(intersection_x) + text_x_offset) if intersection_x > 0 else 10**(np.log10(min(x_plot))+0.1) if len(x_plot)>0 else 100
        text_y = synthetic_rmse + y_range * 0.05 # Offset based on y range
        
        # Ensure text placement is within plot bounds (simple check)
        plot_ymin, plot_ymax = ax.get_ylim()
        text_y = min(text_y, plot_ymax * 0.95) 
        text_y = max(text_y, plot_ymin * 1.05)
        
        plt.annotate(intersection_text, xy=(intersection_x, synthetic_rmse), 
                     xytext=(text_x, text_y), 
                     arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     fontsize=10, ha='left')
    # If no intersection calculated but synthetic RMSE exists, display the comparison text
    elif not pd.isna(synthetic_rmse) and intersection_text != "Intersection could not be determined.":
         plt.text(0.05, 0.95, intersection_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    # Handle case where synthetic RMSE is NaN
    elif pd.isna(synthetic_rmse): 
         plt.text(0.05, 0.95, "Synthetic RMSE calculation failed", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    plt.tight_layout()
    plt.savefig(paths['plot_save_path'], dpi=300)
    logger.info(f"💾 RMSE Plot saved to {paths['plot_save_path']}")
    # plt.show() # Comment out if running as script

    # --- Final Summary (Uses the updated intersection_text) ---
    logger.info("\n--- 📋 RMSE Efficiency Summary ---")
    if not pd.isna(synthetic_rmse): 
        logger.info(f"Synthetic Model RMSE: {synthetic_rmse:.4f}")
    else:
        logger.info("Synthetic Model RMSE: Could not be calculated.")
        
    logger.info("\nAvg RMSE using Real Data Samples:")
    if valid_rmse: # Check if there's real data RMSE to show
        rmse_df = pd.DataFrame({
            'Sample Size': list(real_data_rmse.keys()),
            'Avg RMSE': list(real_data_rmse.values()),
            'Sample %': [n/max_holdout_answers*100 for n in real_data_rmse.keys()] # Calculate % based on keys
        }).sort_values('Sample Size')
        print(rmse_df.to_string(formatters={'Sample Size': '{:,.0f}'.format, 'Avg RMSE': '{:.4f}'.format, 'Sample %': '{:.1f}%'.format}))
    else:
        logger.info("No valid real data RMSEs were calculated.")
    
    # Use the determined intersection_text which is now more robust
    logger.info(f"\nEfficiency Conclusion (RMSE based): {intersection_text}")

if __name__ == "__main__":
    # Expecting the run directory relative path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python evaluate_efficiency_rmse.py <relative_run_directory>")
        print("Example: python evaluate_efficiency_rmse.py results/YYYYMMDD_HHMMSS")
        sys.exit(1)
    run_dir_arg = sys.argv[1]
    main(run_dir_arg) 
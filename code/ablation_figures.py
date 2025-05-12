import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json # For loading metrics if needed

# --- Style Parameters (from example) ---
TAILWIND_COLORS = {
    'slate-50': '#F8FAFC', 'slate-100': '#F1F5F9', 'slate-200': '#E2E8F0',
    'slate-300': '#CBD5E1', 'slate-400': '#94A3B8', 'slate-500': '#64748B',
    'slate-600': '#475569', 'slate-700': '#334155', 'slate-800': '#1E293B',
    'slate-900': '#0F172A',
}
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

# --- Configuration ---
# !!! USER: Please confirm this path is correct for your latest ablation run !!!
ABLATION_RUN_DIR = "/Users/matias/Projects/capstone-new/results_feature_ablation_new_buckets_strategy/20250512_065907_ablation_study_new_buckets" 
FIGURES_OUTPUT_DIR = os.path.join(ABLATION_RUN_DIR, "figures_from_script")
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# RMSE Data from modeling_strategy.md
REAL_DATA_RMSE_POINTS = [
    (254, 1.6807), (1272, 1.6059), (2544, 1.4834), (3816, 1.3666),
    (5088, 1.2475), (6360, 1.1399), (7632, 1.0680), (8904, 1.0025),
    (10176, 0.9188), (11448, 0.8447), (12720, 0.7815), (13992, 0.7191),
    (15264, 0.6631), (16536, 0.6088), (17808, 0.5425), (19080, 0.4766),
    (20352, 0.3960), (21624, 0.3390), (22896, 0.2647), (24168, 0.1821),
    (25440, 0.0185)
]
NN_MODEL_RMSE = 1.1857
NN_EQUIVALENT_ANSWERS = 5818 # Approximate

# --- Helper Functions ---
def load_benchmark_irt(run_dir):
    """Loads the benchmark original 1PL IRT parameters."""
    path = os.path.join(run_dir, "00_benchmark_original_1pl_irt_params.csv")
    if not os.path.exists(path):
        print(f"ERROR: Benchmark IRT file not found at {path}")
        return None
    return pd.read_csv(path)

def load_predicted_irt(run_dir, model_config_name_suffix):
    """Loads predicted 1PL IRT parameters for a given model configuration."""
    # Filename example: 05_predicted_1pl_params_Model_1_Embeddings_Only.csv
    filename = f"05_predicted_1pl_params_{model_config_name_suffix}.csv"
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Predicted IRT file not found for {model_config_name_suffix} at {path}")
        return None
    return pd.read_csv(path)

# --- Plotting Functions ---

def plot_rmse_efficiency_curve(real_data_points, nn_rmse_val, nn_equiv_answers, output_dir):
    """Plots the RMSE efficiency curve."""
    counts = [p[0] for p in real_data_points]
    rmses = [p[1] for p in real_data_points]

    plt.figure(figsize=(10, 6.5))
    # Turn off the grid
    plt.grid(False)
    
    plt.plot(counts, rmses, marker='o', linestyle='-', color=TAILWIND_COLORS['slate-700'], label="Traditional 1PL IRT RMSE")
    
    plt.axhline(y=nn_rmse_val, color=TAILWIND_COLORS['slate-500'], linestyle='--', label=f"NN-derived Difficulty RMSE ({nn_rmse_val:.4f})")
    plt.axvline(x=nn_equiv_answers, color=TAILWIND_COLORS['slate-500'], linestyle=':', 
                label=f"Equivalent to ~23% of all answers")

    # Annotate the intersection point
    plt.scatter([nn_equiv_answers], [nn_rmse_val], s=100, color=TAILWIND_COLORS['slate-800'], zorder=5)
    plt.annotate(f"({nn_equiv_answers}, {nn_rmse_val:.4f})", 
                 (nn_equiv_answers, nn_rmse_val),
                 textcoords="offset points", xytext=(5,15), ha='left', color=TAILWIND_COLORS['slate-800'])

    plt.xlabel("Number of Student Answers Used for Prediction")
    plt.ylabel("RMSE of Difficulty Estimate")
    
    # Update legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if "Traditional 1PL IRT RMSE" in label:
            new_labels.append("Traditional 1PL IRT")
        elif "NN-derived Difficulty RMSE" in label:
            new_labels.append(f"Our model RMSE ({nn_rmse_val:.4f})")
        elif "Equivalent to" in label:
            new_labels.append(label) # Keep this one as is, or rephrase if needed directly in plot call
        else:
            new_labels.append(label)
    plt.legend(handles, new_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_efficiency_curve.png"), dpi=300)
    plt.close()
    print("RMSE efficiency curve plot generated.")

def plot_difficulty_scatter(benchmark_df, predicted_df, model_name_label, output_dir):
    """Creates a scatter plot of benchmark vs. predicted difficulty."""
    if benchmark_df is None or predicted_df is None:
        print(f"Skipping scatter plot for {model_name_label} due to missing data.")
        return

    # Align scales before merging and plotting
    mean_benchmark = benchmark_df['difficulty'].mean()
    std_benchmark = benchmark_df['difficulty'].std()
    mean_predicted = predicted_df['difficulty'].mean()
    std_predicted = predicted_df['difficulty'].std()

    if std_predicted == 0: # Avoid division by zero if predicted difficulties are constant
        print(f"Warning: Predicted difficulties for {model_name_label} have zero standard deviation. Skipping scale alignment for this model, using original values.")
        predicted_df_for_merge = predicted_df.copy()
        # Ensure the column is named 'difficulty_aligned' for consistent merging/plotting
        if 'difficulty' in predicted_df_for_merge.columns and 'difficulty_aligned' not in predicted_df_for_merge.columns:
            predicted_df_for_merge.rename(columns={'difficulty': 'difficulty_aligned'}, inplace=True)
        elif 'difficulty_aligned' not in predicted_df_for_merge.columns:
            # Fallback if 'difficulty' column also doesn't exist, though unlikely
            print(f"Error: Cannot find 'difficulty' or 'difficulty_aligned' in predicted_df for {model_name_label}")
            return 
        predicted_df_for_merge = predicted_df_for_merge # Use this df for merge, already has 'difficulty_aligned'
    else:
        predicted_df_to_align = predicted_df.copy() 
        print(f"DEBUG: Before scaling - Model {model_name_label}:")
        print(f"  Raw predicted mean: {mean_predicted:.2f}, std: {std_predicted:.2f}")
        print(f"  Benchmark mean: {mean_benchmark:.2f}, std: {std_benchmark:.2f}")
        
        aligned_values = ((predicted_df_to_align['difficulty'] - mean_predicted) / std_predicted) * std_benchmark + mean_benchmark
        
        # Create a new DataFrame for merging, containing only question_id and the ALIGNED difficulties
        # Rename the aligned difficulty column to 'difficulty' so merge suffixes apply correctly if it was the only one
        # However, to be explicit and ensure correct column is used for merge, we pass the specific aligned column.
        predicted_df_for_merge = pd.DataFrame({
            'question_id': predicted_df_to_align['question_id'],
            'difficulty_to_merge': aligned_values # Use a distinct name before merge
        })
        
        print(f"DEBUG: After scaling (values prepared for merge) - Model {model_name_label}:")
        print(f"  Prepared predicted mean: {predicted_df_for_merge['difficulty_to_merge'].mean():.2f}, std: {predicted_df_for_merge['difficulty_to_merge'].std():.2f}")

    # Pass this new df to the merge, explicitly naming columns if necessary to avoid suffix collision with original 'difficulty'
    # Or ensure predicted_df_for_merge only has 'question_id' and the column to be suffixed.
    # Benchmark: difficulty -> difficulty_benchmark
    # Predicted: difficulty_to_merge -> difficulty_predicted_aligned (if 'difficulty' name was used in predicted_df_for_merge)
    merged_df = pd.merge(benchmark_df.rename(columns={'difficulty': 'difficulty_benchmark'}), 
                         predicted_df_for_merge.rename(columns={'difficulty_to_merge': 'difficulty_predicted_aligned'}), 
                         on="question_id")
    
    # DEBUG: Print ranges before plotting
    print(f"--- Debugging plot_difficulty_scatter for: {model_name_label} (after merge) ---")
    print(f"Benchmark difficulty range: {merged_df['difficulty_benchmark'].min():.2f} to {merged_df['difficulty_benchmark'].max():.2f}")
    print(f"Predicted (aligned) difficulty range in merged_df: {merged_df['difficulty_predicted_aligned'].min():.2f} to {merged_df['difficulty_predicted_aligned'].max():.2f}")
    print(f"Mean predicted (aligned) in merged_df: {merged_df['difficulty_predicted_aligned'].mean():.2f}, Std Dev predicted (aligned) in merged_df: {merged_df['difficulty_predicted_aligned'].std():.2f}")
    print(f"Mean benchmark: {merged_df['difficulty_benchmark'].mean():.2f}, Std Dev benchmark: {merged_df['difficulty_benchmark'].std():.2f}")
    print(f"--- End Debugging ---")

    plt.figure(figsize=(8, 8))
    # Turn off the grid
    plt.grid(False)
    
    plt.scatter(merged_df['difficulty_benchmark'], merged_df['difficulty_predicted_aligned'], 
                alpha=0.6, color=TAILWIND_COLORS['slate-500'], 
                edgecolor=TAILWIND_COLORS['slate-700'], s=50)
    
    min_val = min(merged_df['difficulty_benchmark'].min(), merged_df['difficulty_predicted_aligned'].min())
    max_val = max(merged_df['difficulty_benchmark'].max(), merged_df['difficulty_predicted_aligned'].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color=TAILWIND_COLORS['slate-400'], alpha=0.7, label="y=x line")

    # Trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['difficulty_benchmark'], merged_df['difficulty_predicted_aligned'])
    x_trend = np.array([min_val, max_val])
    y_trend = slope * x_trend + intercept
    plt.plot(x_trend, y_trend, color=TAILWIND_COLORS['slate-800'], linestyle='-', linewidth=2, label="Trendline")

    # Correlations for annotation (Pearson on aligned, Spearman can be on original or aligned - should be similar)
    # For consistency, let's use aligned for Pearson. Spearman is rank-based, less affected by linear scale changes.
    pearson_r, _ = stats.pearsonr(merged_df['difficulty_benchmark'], merged_df['difficulty_predicted_aligned'])
    spearman_r, _ = stats.spearmanr(merged_df['difficulty_benchmark'], merged_df['difficulty_predicted_aligned']) # Use aligned for Spearman too from merged_df
    
    plt.annotate(f"Pearson r = {pearson_r:.4f}\nSpearman ρ = {spearman_r:.4f}\nN = {len(merged_df)}",
                 (0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="w", ec="gray", alpha=0.8))

    plt.xlabel("Actual Difficulty")
    plt.ylabel("Predicted Difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"difficulty_scatter_{model_name_label.replace(' ', '_').replace('(','').replace(')','')}.png"), dpi=300)
    plt.close()
    print(f"Difficulty scatter plot for {model_name_label} generated.")

def plot_difficulty_density(benchmark_df, predicted_df, model_name_label, output_dir):
    """Plots KDEs for benchmark and predicted difficulties."""
    if benchmark_df is None or predicted_df is None:
        print(f"Skipping density plot for {model_name_label} due to missing data.")
        return
        
    benchmark_diff = benchmark_df['difficulty']
    
    # Align scales for predicted difficulties
    mean_benchmark = benchmark_diff.mean()
    std_benchmark = benchmark_diff.std()
    mean_predicted = predicted_df['difficulty'].mean()
    std_predicted = predicted_df['difficulty'].std()

    if std_predicted == 0:
        print(f"Warning: Predicted difficulties for {model_name_label} have zero standard deviation. Plotting original predicted values.")
        predicted_diff_aligned = predicted_df['difficulty']
    else:
        predicted_diff_aligned = ((predicted_df['difficulty'] - mean_predicted) / std_predicted) * std_benchmark + mean_benchmark

    plt.figure(figsize=(10, 6))
    plt.grid(False)
    sns.kdeplot(benchmark_diff, fill=True, color=TAILWIND_COLORS['slate-700'], alpha=0.5, label='Actual Difficulty')
    sns.kdeplot(predicted_diff_aligned, fill=True, color=TAILWIND_COLORS['slate-500'], alpha=0.5, label='Predicted Difficulty')
    
    plt.xlabel("1PL Difficulty")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"difficulty_density_{model_name_label.replace(' ', '_').replace('(','').replace(')','')}.png"), dpi=300)
    plt.close()
    print(f"Difficulty density plot for {model_name_label} generated.")

def plot_difficulty_ridgeline(benchmark_df, predicted_dfs_all_models, output_dir):
    """Creates a ridgeline plot of difficulties for benchmark and all ablation models."""
    if benchmark_df is None or not predicted_dfs_all_models:
        print("Skipping ridgeline plot due to missing data.")
        return

    plt.figure(figsize=(12, 8)) 
    
    mean_benchmark = benchmark_df['difficulty'].mean()
    std_benchmark = benchmark_df['difficulty'].std()

    # Define the desired order and display names for models
    model_name_map = {
        "Benchmark": "Benchmark", # Add Benchmark here for ordering
        "Model_1_Embeddings_Only": "Model 1 (Emb. Only)",
        "Model_2_Embeddings_QuestionFeatures": "Model 2 (+QFeat)",
        "Model_3_Embeddings_Question_OptionFeatures": "Model 3 (+OptFeat)",
        "Model_4_Embeddings_Question_Option_LLMFeatures": "Model 4 (+LLMFeat)"
    }
    # This model_order will define the plotting sequence and categorical order
    model_plot_order = [
        "Benchmark",
        "Model_1_Embeddings_Only",
        "Model_2_Embeddings_QuestionFeatures",
        "Model_3_Embeddings_Question_OptionFeatures",
        "Model_4_Embeddings_Question_Option_LLMFeatures"
    ]

    plot_data_list = []
    if benchmark_df is not None:
        temp_df = pd.DataFrame() # Create new DataFrame
        temp_df['Difficulty'] = benchmark_df['difficulty'] # Use 'Difficulty' consistently
        temp_df['Model'] = model_name_map['Benchmark']
        plot_data_list.append(temp_df)

    # Iterate through model suffixes used in predicted_dfs_all_models keys
    # (which are like "Model_1_Embeddings_Only", etc.)
    for model_key_suffix in [m for m in model_plot_order if m != "Benchmark"]:
        if model_key_suffix in predicted_dfs_all_models and predicted_dfs_all_models[model_key_suffix] is not None:
            pred_df_current = predicted_dfs_all_models[model_key_suffix]
            
            temp_df = pd.DataFrame() # Create new DataFrame for aligned data
            mean_predicted_current = pred_df_current['difficulty'].mean()
            std_predicted_current = pred_df_current['difficulty'].std()
            
            if std_predicted_current == 0:
                print(f"Warning: Predicted difficulties for {model_key_suffix} have zero std dev. Using original values for ridgeline.")
                temp_df['Difficulty'] = pd.Series(dtype=float) # empty series
            else:
                temp_df['Difficulty'] = ((pred_df_current['difficulty'] - mean_predicted_current) / std_predicted_current) * std_benchmark + mean_benchmark
            
            temp_df['Model'] = model_name_map.get(model_key_suffix, model_key_suffix) 
            plot_data_list.append(temp_df)
        else:
            print(f"Warning: Predicted data for {model_key_suffix} not found for ridgeline plot.")

    if not plot_data_list:
        print("No data available for ridgeline plot after filtering.")
        return

    combined_plot_df = pd.concat(plot_data_list)
    combined_plot_df.rename(columns={'Difficulty': 'Difficulty'}, inplace=True)

    # Using seaborn's FacetGrid for a ridgeline-like plot
    try:
        # Ensure 'Model' is treated as categorical and in the desired order for plotting
        display_names_ordered = [model_name_map[m] for m in model_plot_order]
        combined_plot_df['Model'] = pd.Categorical(combined_plot_df['Model'], categories=display_names_ordered, ordered=True)
        
        palette = sns.color_palette("viridis_r", n_colors=len(display_names_ordered))
        # Reverse the order for plotting bottom-up in FacetGrid
        g = sns.FacetGrid(combined_plot_df, row="Model", hue="Model", aspect=7, height=1, palette=palette, row_order=list(reversed(display_names_ordered)))

        g.map(sns.kdeplot, "Difficulty", fill=True, alpha=0.7, lw=1.5, bw_adjust=0.5, cut=0)
        g.map(sns.kdeplot, "Difficulty", color="white", lw=2, bw_adjust=0.5, cut=0)
        
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0.02, .1, label, fontweight="bold", color='black', # Ensure label is readable
                    ha="left", va="center", transform=ax.transAxes, fontsize=10)

        g.map(label, "Difficulty")
        g.fig.subplots_adjust(hspace=-0.7) # Overlap the plots more
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        g.set_xlabels("Difficulty")
        plt.yticks([])
        plt.xlabel("Difficulty")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

    except Exception as e:
        print(f"Error creating ridgeline plot with FacetGrid: {e}. Check data or try alternative.")
        plt.close() # Close the potentially broken plot
        return

    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust for suptitle & legend if added externally
    plt.savefig(os.path.join(output_dir, "difficulty_ridgeline_ablation.png"), dpi=300)
    plt.close()
    print("Difficulty ridgeline plot generated.")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Looking for ablation results in: {ABLATION_RUN_DIR}")

    benchmark_irt_df = load_benchmark_irt(ABLATION_RUN_DIR)
    
    model_suffixes = {
        "Model_1_Embeddings_Only": "Model_1_Embeddings_Only",
        "Model_2_Embeddings_QuestionFeatures": "Model_2_Embeddings_QuestionFeatures",
        "Model_3_Embeddings_Question_OptionFeatures": "Model_3_Embeddings_Question_OptionFeatures",
        "Model_4_Embeddings_Question_Option_LLMFeatures": "Model_4_Embeddings_Question_Option_LLMFeatures"
    }
    
    predicted_irt_dfs = {}
    for key_in_dict, suffix_for_file in model_suffixes.items(): 
        predicted_irt_dfs[key_in_dict] = load_predicted_irt(ABLATION_RUN_DIR, suffix_for_file)

    if benchmark_irt_df is None or predicted_irt_dfs.get("Model_4_Embeddings_Question_Option_LLMFeatures") is None:
        print("Critical data files missing. Some plots may not be generated.")
    
    # 1. RMSE Efficiency Curve
    plot_rmse_efficiency_curve(REAL_DATA_RMSE_POINTS, NN_MODEL_RMSE, NN_EQUIVALENT_ANSWERS, FIGURES_OUTPUT_DIR)

    # 2. Scatter plot for Model 4
    model4_suffix = "Model_4_Embeddings_Question_Option_LLMFeatures"
    if predicted_irt_dfs.get(model4_suffix) is not None:
        plot_difficulty_scatter(benchmark_irt_df, predicted_irt_dfs[model4_suffix], 
                                "Model 4 (Full)", FIGURES_OUTPUT_DIR)

    # 3. Density plot for Model 4
    if predicted_irt_dfs.get(model4_suffix) is not None:
        plot_difficulty_density(benchmark_irt_df, predicted_irt_dfs[model4_suffix], 
                                "Model 4 (Full)", FIGURES_OUTPUT_DIR)
    
    # 4. Ridgeline plot for all models
    plot_difficulty_ridgeline(benchmark_irt_df, predicted_irt_dfs, FIGURES_OUTPUT_DIR)

    print(f"All plots saved to: {FIGURES_OUTPUT_DIR}") 
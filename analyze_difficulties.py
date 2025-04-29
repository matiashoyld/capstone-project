import pandas as pd
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python analyze_difficulties.py <results_directory>")
    sys.exit(1)

run_dir = sys.argv[1]
# file_orig = os.path.join(run_dir, '01_difficulty.csv') # Old filename
file_orig = os.path.join(run_dir, '01_irt_2pl_params.csv') # New filename
file_pred = os.path.join(run_dir, '05_predicted_2pl_params.csv') # New filename

if not os.path.exists(file_orig):
    print(f"Error: File not found - {file_orig}")
    sys.exit(1)
if not os.path.exists(file_pred):
    print(f"Error: File not found - {file_pred}")
    sys.exit(1)

print(f'Loading {file_orig}')
df_orig = pd.read_csv(file_orig)
print(f'Loading {file_pred}')
df_pred = pd.read_csv(file_pred)

print('\n--- Original Difficulty Stats ---')
print(df_orig['difficulty'].describe())

print('\n--- Predicted Difficulty Stats ---')
print(df_pred['difficulty'].describe())

print('\n--- Merging and Checking Correlation ---')
merged_df = pd.merge(df_orig, df_pred, on='question_id', suffixes=['_orig', '_pred'])
print(f'Merged shape: {merged_df.shape}')

if not merged_df.empty:
    corr = merged_df['difficulty_orig'].corr(merged_df['difficulty_pred'])
    print(f'Pearson Correlation: {corr:.4f}')
else:
    print('Merged DataFrame is empty.')

print("\nAnalysis complete.") 
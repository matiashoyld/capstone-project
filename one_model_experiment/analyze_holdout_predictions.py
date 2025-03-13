import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_predictions():
    # Load the probability matrix, binary matrix and holdout questions
    proba = pd.read_csv('predictions/08_holdout_probability_matrix.csv', index_col=0)
    binary = pd.read_csv('predictions/08_holdout_binary_prediction_matrix.csv', index_col=0)
    holdout = pd.read_csv('data/07_holdout_test_questions.csv')
    
    # Convert question_id to string in holdout dataframe
    holdout['question_id'] = holdout['question_id'].astype(str)
    
    # Create a mapping from question_id to is_correct and irt_difficulty
    holdout_q_correct = holdout.set_index('question_id')['is_correct']
    holdout_q_difficulty = holdout.set_index('question_id')['irt_difficulty']
    
    print(f"Binary matrix shape: {binary.shape}")
    print(f"Number of holdout questions: {len(holdout)}")
    
    # Check the distribution of probabilities
    all_probs = proba.values.flatten()
    print(f"\nProbability distribution:")
    print(f"Mean: {np.mean(all_probs):.4f}")
    print(f"Median: {np.median(all_probs):.4f}")
    print(f"Min: {np.min(all_probs):.4f}")
    print(f"Max: {np.max(all_probs):.4f}")
    print(f"Std dev: {np.std(all_probs):.4f}")
    
    # Check the distribution of binary predictions
    all_binary = binary.values.flatten()
    print(f"\nBinary prediction distribution:")
    print(f"Percentage of 1s: {np.mean(all_binary) * 100:.2f}%")
    print(f"Percentage of 0s: {(1 - np.mean(all_binary)) * 100:.2f}%")
    
    # Calculate correlation between predictions and correctness
    avg_pred_by_question = binary.mean(axis=0)
    correctness_corr = pd.DataFrame({
        'avg_prediction': avg_pred_by_question,
        'actual_correctness': holdout_q_correct
    })
    
    # Calculate correlation
    correlation = correctness_corr['avg_prediction'].corr(correctness_corr['actual_correctness'])
    print(f"\nCorrelation between average predictions and actual correctness: {correlation:.4f}")
    
    # Calculate correlation between predictions and difficulty
    difficulty_corr = pd.DataFrame({
        'avg_prediction': avg_pred_by_question,
        'irt_difficulty': holdout_q_difficulty
    })
    
    # Calculate correlation
    difficulty_correlation = difficulty_corr['avg_prediction'].corr(difficulty_corr['irt_difficulty'])
    print(f"\nCorrelation between average predictions and IRT difficulty: {difficulty_correlation:.4f}")
    
    # Create scatter plot of average predictions vs actual correctness
    plt.figure(figsize=(10, 6))
    plt.scatter(correctness_corr['actual_correctness'], correctness_corr['avg_prediction'], alpha=0.6)
    plt.title('Average Prediction vs Actual Correctness Rate')
    plt.xlabel('Actual Correctness Rate')
    plt.ylabel('Average Prediction (across users)')
    plt.grid(alpha=0.3)
    
    # Add a diagonal line for reference
    plt.plot([0, 1], [0, 1], 'r--')
    
    # Calculate and add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        correctness_corr['actual_correctness'],
        correctness_corr['avg_prediction']
    )
    plt.plot(
        correctness_corr['actual_correctness'],
        intercept + slope * correctness_corr['actual_correctness'],
        'g-',
        label=f'Regression line (r={r_value:.2f})'
    )
    plt.legend()
    
    # Save the plot
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/08_prediction_vs_correctness.png')
    plt.close()
    
    # Create scatter plot of average predictions vs IRT difficulty
    plt.figure(figsize=(10, 6))
    plt.scatter(difficulty_corr['irt_difficulty'], difficulty_corr['avg_prediction'], alpha=0.6)
    plt.title('Average Prediction vs IRT Difficulty')
    plt.xlabel('IRT Difficulty')
    plt.ylabel('Average Prediction (across users)')
    plt.grid(alpha=0.3)
    
    # Calculate and add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        difficulty_corr['irt_difficulty'],
        difficulty_corr['avg_prediction']
    )
    plt.plot(
        difficulty_corr['irt_difficulty'],
        intercept + slope * difficulty_corr['irt_difficulty'],
        'g-',
        label=f'Regression line (r={r_value:.2f})'
    )
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/08_prediction_vs_difficulty.png')
    plt.close()
    
    # Create distribution plot of probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(all_probs, bins=50, kde=True)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.axvline(0.5, color='r', linestyle='--', label='Decision threshold (0.5)')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/08_probability_distribution.png')
    plt.close()
    
    # Create histogram of actual correctness rates
    plt.figure(figsize=(10, 6))
    sns.histplot(holdout_q_correct, bins=20, kde=True)
    plt.title('Distribution of Actual Correctness Rates')
    plt.xlabel('Correctness Rate')
    plt.ylabel('Count')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/08_correctness_distribution.png')
    plt.close()
    
    # Compare predictions by correctness bins
    holdout['pred_correctness'] = holdout['question_id'].map(avg_pred_by_question)
    
    # Create boxplot of predictions by correctness bin
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='correctness_bin', y='pred_correctness', data=holdout)
    plt.title('Predictions by Correctness Bin')
    plt.xlabel('Correctness Bin')
    plt.ylabel('Average Prediction')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/08_predictions_by_correctness_bin.png')
    plt.close()
    
    # Compare predictions by difficulty bins
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='difficulty_bin', y='pred_correctness', data=holdout)
    plt.title('Predictions by Difficulty Bin')
    plt.xlabel('Difficulty Bin')
    plt.ylabel('Average Prediction')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/08_predictions_by_difficulty_bin.png')
    plt.close()
    
    print("\nAnalysis complete. Visualizations saved to 'figures/' directory.")

if __name__ == "__main__":
    analyze_predictions() 
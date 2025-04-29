# Paper Data for Neural Network IRT Estimation

This directory contains the key data files needed for creating visualizations for the research paper on neural network-based Item Response Theory (IRT) parameter estimation.

## Data Files

### IRT Difficulty Comparison Data

- `top100_difficulty_comparison.csv`: Comparison between original and estimated IRT difficulties using the top 100 users
- `all_users_difficulty_comparison.csv`: Comparison between original and estimated IRT difficulties using all users (1,867)

These files contain:
- `question_id`: Identifier for the question
- `estimated_difficulty`: Difficulty parameter estimated by our model
- `original_difficulty`: Original IRT difficulty parameter
- `discrimination`: Discrimination parameter estimated by our model
- `discrepancy`: Absolute difference between estimated and original difficulty

### Performance Metrics

- `performance_metrics_comparison.csv`: Comparison of key performance metrics between the top 100 users and all users approaches
- `top100_irt_metrics.csv`: Detailed metrics for the top 100 users approach
- `all_users_irt_metrics.csv`: Detailed metrics for the all users approach

These files contain metrics such as:
- Pearson correlation
- Spearman correlation
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

### Prediction Matrices

- `top100_probability_matrix.csv`: Probability predictions for 100 users × 469 questions
- `all_users_probability_matrix.csv`: Probability predictions for 1,867 users × 469 questions

These matrices have:
- Rows: User IDs
- Columns: Question IDs
- Values: Probability of correct answer

### User Abilities Data

- `top100_user_abilities.csv`: Ability parameters estimated for the top 100 users
- `all_users_abilities.csv`: Ability parameters estimated for all 1,867 users

### Question Data

- `holdout_questions.csv`: Original data for the 469 holdout questions, including:
  - `question_id`: Identifier for the question
  - `is_correct`: Average correctness rate
  - `irt_difficulty`: Original IRT difficulty
  - `correctness_bin`: Bin for stratification based on correctness
  - `difficulty_bin`: Bin for stratification based on difficulty

### IRT Parameter Estimates

- `top100_irt_difficulties.csv`: Difficulty and discrimination parameters estimated using top 100 users
- `all_users_irt_difficulties.csv`: Difficulty and discrimination parameters estimated using all users

### Summary Statistics

- `prediction_variation_stats.csv`: Statistics about prediction variations across users and questions, including:
  - Standard deviation across questions
  - Standard deviation across users
  - Mean, min, and max probabilities
  - Number of unique prediction values

## Visualizations

The `visualizations/` directory contains pre-generated figures:

- `top100_difficulty_comparison.png`: Scatter plot comparing original vs. estimated difficulties (top 100 users)
- `all_users_difficulty_comparison.png`: Scatter plot comparing original vs. estimated difficulties (all users)
- `correlation_comparison.png`: Bar chart comparing correlation metrics
- `error_comparison.png`: Bar chart comparing error metrics

## Usage for Paper

When creating visualizations for your paper, consider focusing on:

1. **Comparison of Original vs. Estimated Difficulties**: Shows how well the model estimates IRT parameters
2. **Comparison Between Top 100 Users and All Users Approaches**: Demonstrates robustness and efficiency
3. **Prediction Variation Analysis**: Illustrates the balanced influence of user and question features
4. **User Ability Distribution**: Shows the model's ability to identify different user proficiency levels

For specific visualization scripts or additional data exports, please refer to the source code in the main project directory. 
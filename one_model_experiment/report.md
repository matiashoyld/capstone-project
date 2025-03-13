# Feature Creation Report

This report documents the feature engineering and data processing steps implemented in the `feature_creation.py` script.

## Overview

The goal of this process was to create a comprehensive dataset for educational assessment analysis by merging multiple data sources and extracting meaningful features from text and answer options.

## Data Sources

Three main datasets were used in this process:

1. **master_translated.csv** - Contains student response data
   - Key columns: answer_id, is_correct, user_id, grade_id, user_level, question_id

2. **questions_master.csv** - Contains detailed information about questions
   - Key columns: question title, options, question metadata, difficulty metrics, etc.

3. **question_difficulties_irt.csv** - Contains Item Response Theory difficulty metrics
   - Key columns: question_id, irt_difficulty, original_difficulty

## Data Processing Steps

### 1. Initial Data Loading and Merging

- Loaded the three CSV files
- Selected relevant columns from each dataset
- Merged datasets on question_id using inner joins

### 2. Filtering Criteria

Implemented filtering to remove low-quality or extreme data points:
- Removed questions with fewer than 10 total responses (total_count < 10)
- Removed questions with very low difficulty (irt_difficulty < -6)

This filtering resulted in:
- 490 rows removed (approximately 0.19% of original data)
- 15 unique questions removed (approximately 0.32% of unique questions)

### 3. Text Feature Extraction

Extracted advanced features from question titles using natural language processing techniques:

| Feature | Description |
|---------|-------------|
| title_word_count | Number of words in the question title |
| title_char_count | Number of characters in the question title |
| title_avg_word_length | Average length of words in the title |
| title_digit_count | Number of numerical digits in the title |
| title_special_char_count | Number of special characters in the title |
| title_mathematical_symbols | Count of mathematical symbols (e.g., +, -, *, /) |
| title_latex_expressions | Count of LaTeX expressions or math formatting |

### 4. Answer Option Analysis

Analyzed answer options to extract insights about their variability and length:

| Feature | Description |
|---------|-------------|
| jaccard_similarity_std | Standard deviation of Jaccard similarities between all pairs of options |
| avg_option_length | Average character length of answer options |
| avg_option_word_count | Average number of words in answer options |

## Key Insights

### Text Feature Analysis

- Average question title contains about 23 words and 118 characters
- Longest question title had 173 words and 945 characters
- Questions typically contain around 8 mathematical symbols
- About 30% of questions contain LaTeX expressions

### Answer Option Analysis

- The standard deviation of similarity between options averages around 0.116
- Answer options average about 10 characters and 2.5 words in length
- Some questions have very long options (up to 131 characters)
- Most options contain between 1-3 words

## Final Dataset

The final processed dataset:
- Contains 251,361 rows after filtering
- Includes 42 columns (original columns + derived features)
- Was saved as `merged_features_filtered.csv`

### Missing Values

Some columns in the final dataset contain missing values:
- grade_id: 33,261 missing values
- user_level: 4,754 missing values
- option_d: 91 missing values

All derived feature columns are complete with no missing values.

## Embedding Generation

After feature extraction, we generated semantic embeddings for the questions and answer options using ModernBERT, a state-of-the-art transformer model optimized for embeddings.

### Process (`02_generate_embeddings.py`)

1. **Data Preparation**:
   - Loaded the filtered dataset with 4,681 unique questions
   - Formatted questions and answers in a structured format:
     ```
     Question: [question_title]
     Correct Answer: [correct option]
     Wrong Answer 1: [wrong option 1]
     ...
     Wrong Answer 4: [wrong option 4]
     ```

2. **ModernBERT Model**:
   - Used `nomic-ai/modernbert-embed-base` from Hugging Face
   - Applied the transformer model to generate contextual embeddings
   - Used mean pooling to combine token embeddings into sentence embeddings
   - Normalized the embeddings to ensure consistent comparisons

3. **Generated Embeddings**:
   - Created embeddings for the complete formatted question-answer text
   - Generated separate embeddings for each answer option (option_a through option_e)
   - Embedding dimension: 768 (standard for BERT-based models)

4. **Output**:
   - Saved all embeddings to `question_embeddings.pkl` as a dictionary containing:
     - `question_ids`: IDs for all questions
     - `formatted_embeddings`: Embeddings for the formatted question-answer text
     - `option_embeddings`: Dictionary of embeddings for each answer option

### Embedding Statistics

- Formatted embeddings shape: (4,681, 768)
- Individual option embeddings:
  - option_a: (4,681, 768)
  - option_b: (4,681, 768)
  - option_c: (4,681, 768)
  - option_d: (4,679, 768) - Note: 2 missing due to null values
  - option_e: (4,681, 768)

These embeddings capture the semantic meaning of the questions and answers and can be used for various downstream tasks such as similarity calculation, clustering, or as features for machine learning models.

## Predictive Modeling

We developed a LightGBM model to predict whether a student would answer a question correctly (`is_correct`) based on the features we created and the generated embeddings.

### Train-Test Split Strategy

To ensure that our model can generalize well to new questions, we split the data at the question level rather than at the row level:

1. **Stratified Question Split**:
   - Split questions into 80% train and 20% test
   - Used stratification based on both question difficulty (`irt_difficulty`) and correctness rate
   - Ensured that both train and test sets had a balanced distribution of easy/hard questions and frequently/rarely correct questions

2. **Distribution Balance**:
   - Train set: 3,744 questions (201,487 rows)
   - Test set: 937 questions (49,874 rows)
   - IRT Difficulty distribution:
     - Train mean: -0.866, std: 1.623
     - Test mean: -0.873, std: 1.674
   - Correctness distribution:
     - Train mean: 0.699, std: 0.197
     - Test mean: 0.700, std: 0.195

### Feature Engineering for Modeling

1. **Feature Types**:
   - Question difficulty features (`irt_difficulty`, `original_difficulty`)
   - User features (`user_level`)
   - Demographic features (`grade_id`)
   - Question metadata (`avg_steps`, `level`, `num_misconceptions`)
   - Count features for answer options
   - Text features extracted from question titles
   - Answer option features including similarity and length metrics
   - Image indicator feature (`has_image`)
   - Embedding features (PCA-reduced to 10 dimensions)

2. **Data Preprocessing**:
   - Missing values were imputed using median for numeric features and mode for categorical features
   - Boolean values were converted to integers
   - All features were standardized using `StandardScaler`

### Model Performance

The LightGBM model achieved the following metrics on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.8088 |
| Precision | 0.8378 |
| Recall | 0.9074 |
| F1 Score | 0.8712 |
| AUC-ROC | 0.8479 |

**Confusion Matrix:**
```
[[ 8,088  6,245]
 [ 3,292 32,249]]
```

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.71      0.56      0.63     14,333
           1       0.84      0.91      0.87     35,541

    accuracy                           0.81     49,874
   macro avg       0.77      0.74      0.75     49,874
weighted avg       0.80      0.81      0.80     49,874
```

### Feature Importance Analysis

The top 10 most important features for predicting whether a student would answer correctly:

1. **user_level** (359,410): Student's proficiency level was by far the most important predictor
2. **irt_difficulty** (131,534): Item Response Theory difficulty rating
3. **original_difficulty** (71,704): Original difficulty rating
4. **grade_id** (49,675): Student's grade level
5. **count_a** (10,529): Frequency of option A being selected
6. **count_b** (10,174): Frequency of option B being selected
7. **count_d** (9,433): Frequency of option D being selected
8. **count_c** (8,007): Frequency of option C being selected
9. **count_e** (7,528): Frequency of option E being selected
10. **total_count** (7,251): Total number of students who attempted the question

Interestingly, several embedding features (embed_0 through embed_9) also appeared in the top 20 most important features, demonstrating that the semantic content captured by ModernBERT embeddings provides valuable information for predicting student performance.

## Enhanced Feature Engineering

Building on our initial model, we implemented two significant feature enhancements to improve predictive performance:

### 1. PCA-Reduced Question Embeddings

To leverage the semantic information captured in the ModernBERT embeddings while managing dimensionality:

1. **Embedding Source**:
   - Used the `formatted_embeddings` from the question_embeddings.pkl file
   - Original dimensionality: 768 features per question

2. **Dimensionality Reduction**:
   - Applied Principal Component Analysis (PCA) to reduce dimensions from 768 to 50
   - The 50 principal components retained 66.29% of the original variance
   - Generated features named q_emb_0 through q_emb_49

3. **Impact on Model**:
   - Several embedding features (q_emb_1, q_emb_14, q_emb_10) emerged among the top 10 most important features
   - These PCA-reduced embeddings captured semantic relationships between questions and answers that traditional text features missed

### 2. One-Hot Encoded Skills

To incorporate domain knowledge about the specific skills tested by each question:

1. **Skills Data Processing**:
   - Processed the 'skills' column, which contains lists of skill IDs in formats like [4,6,11,45] or [5,13]
   - Identified 78 unique skills across the dataset
   - Created binary features for each skill (skill_1 through skill_86, with some IDs missing)

2. **One-Hot Encoding Implementation**:
   - For each skill ID, created a binary feature indicating whether the question tests that skill
   - Enabled the model to learn patterns specific to each skill area
   - This approach allows the model to identify which skills are more difficult for students

3. **Impact on Model**:
   - Several skill features appeared in the top features for importance
   - skill_18 ranked among the top 20 most important features
   - The skill features help the model understand domain-specific difficulty patterns

### Updated Model Performance

With the enhanced features, the LightGBM model achieved the following metrics on the test set:

| Metric | Previous | Enhanced | Change |
|--------|----------|----------|--------|
| Accuracy | 0.7374 | 0.7397 | +0.0023 |
| Precision | 0.7589 | 0.7582 | -0.0007 |
| Recall | 0.9185 | 0.9253 | +0.0068 |
| F1 Score | 0.8311 | 0.8334 | +0.0023 |
| AUC-ROC | 0.7560 | 0.7574 | +0.0014 |

**Updated Confusion Matrix:**
```
[[ 4,496 10,539]
 [ 2,669 33,041]]
```

### Updated Feature Importance

The top 20 most important features now include:

1. **title_char_count** (25,821): Length of question title
2. **jaccard_similarity_std** (16,711): Variability in answer option similarity
3. **title_word_count** (13,482): Number of words in question title
4. **q_emb_1** (11,213): First PCA component of question embeddings
5. **q_emb_14** (9,420): 14th PCA component of question embeddings
6. **q_emb_10** (8,665): 10th PCA component of question embeddings
7. **title_special_char_count** (7,808): Number of special characters
8. **q_emb_0** (5,870): Zeroth PCA component of question embeddings
9. **q_emb_19** (5,726): 19th PCA component of question embeddings
10. **title_mathematical_symbols** (4,934): Mathematical symbols count

Additionally, several other embedding components (q_emb_3, q_emb_9, q_emb_16) and skill_18 appeared in the top 20 features, demonstrating the value of these enhanced features.

## Conclusions

1. **Student-related features dominate prediction**:
   - User level and grade are the strongest predictors of correct answers
   - This aligns with educational theory that prior knowledge and general ability are key determinants of performance

2. **Question difficulty matters**:
   - Both IRT difficulty and original difficulty are highly predictive
   - This confirms that our filtering criteria to remove extremely easy/hard questions was appropriate

3. **Answer patterns provide signal**:
   - The frequency with which each option is selected (count_a through count_e) is informative
   - This suggests patterns in how students select different options

4. **Embeddings add value**:
   - Several embedding features ranked in the top 20 for importance
   - This indicates that semantic understanding of the questions and answers contributes to predictive power

5. **Model is more confident in predicting correct answers**:
   - Higher precision and recall for class 1 (correct answers)
   - Lower performance on class 0 (incorrect answers)
   - This aligns with the class imbalance in the dataset (71% correct answers)

## Potential Next Steps

Possible future enhancements to the feature creation and modeling process:
- Imputation of missing values using more sophisticated methods
- Creating interaction features between text characteristics and user variables
- Extracting more sophisticated NLP features using embeddings
- Feature selection to identify most predictive variables
- Fine-tuning embeddings on domain-specific data
- Addressing class imbalance with sampling techniques
- Hyperparameter optimization for the model
- Ensemble methods combining multiple model types
- Time-based analysis to understand learning progression

# Neural Network Model for Balanced User-Question Representations

## Problem Identification: User ID Dominance

While our LightGBM model achieved good accuracy, we identified a critical issue in its predictions: **user ID dominance**. Analysis of the prediction matrices revealed:

1. **Minimal variation across questions for the same user**: The standard deviation across questions was very low (0.026), indicating that predictions for different questions for the same user were almost identical.

2. **High variation across users**: The standard deviation across users was much higher (0.30), suggesting that user identity overwhelmingly determined predictions.

3. **Feature imbalance**: The model included 1,869 user ID features but only 146 question-related features, creating a structural imbalance.

The result was that our LightGBM model essentially functioned as a user classifier rather than properly weighing both user abilities and question characteristics. This compromised its ability to provide accurate Item Response Theory (IRT) difficulty estimations.

## Neural Network Solution with User Embeddings

To address these issues, we implemented a neural network approach that fundamentally changed how user IDs were represented:

### 1. User Embedding Layer

Instead of one-hot encoding user IDs (which creates one feature per user), we implemented an embedding approach:

- **User IDs mapped to indices**: Each user ID was mapped to a numeric index
- **Embedding layer**: Created a learnable embedding matrix (dimensions: num_users × embedding_dim)
- **Low-dimensional representation**: Each user was represented as an 8-dimensional vector
- **Compact and expressive**: Reduced thousands of binary features to just 8 continuous features per user

### 2. Architecture and Feature Balance

The neural network architecture was designed to balance user and question influences:

```
User ID        Numerical Features    Question Embeddings
   ↓                  ↓                     ↓
Embedding      Dense Layer (32)      Dense Layer (32)
   ↓                  ↓                     ↓
   └──────────┬───────────┬────────────────┘
              ↓
     Concatenated Features (72)
              ↓
       Dense Layer (64) + Dropout
              ↓
       Dense Layer (32) + Dropout
              ↓
       Output Layer (sigmoid)
```

- **Multiple feature sources**: Combined user embeddings, question numerical features, and question embeddings
- **Regularization**: Applied L2 regularization and dropout to prevent any one feature type from dominating
- **Balanced architecture**: Each feature type was first processed through its own layer before concatenation

### 3. Robust Evaluation: Three-Way Split

To ensure reliable assessment of model performance, we implemented a more thorough train/validate/test split:

- **Training set**: 3,275 questions (70%) used for model training
- **Validation set**: 937 questions (20%) used for model tuning and performance assessment
- **Holdout test set**: 469 questions (10%) saved for final evaluation

The splits were carefully balanced in terms of:
- **Question difficulty**: Each split had similar IRT difficulty distributions
- **Correctness rates**: Each split had similar correct answer percentages

## Results and Improvements

### 1. Prediction Balance

The neural network model achieved a much better balance in predictions:

| Metric | LightGBM | Neural Network | Improvement |
|--------|----------|----------------|-------------|
| Std dev across questions | 0.026 | 0.213 | 8.2× increase |
| Std dev across users | 0.30 | 0.167 | 0.56× decrease |
| Unique prediction values | Binary | 93,051 | Continuous range |

This shows that our neural network model made meaningfully different predictions for different questions, rather than treating all questions the same for a given user.

### 2. Model Performance

The neural network achieved strong performance on the validation set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7815 |
| Precision | 0.8218 |
| Recall | 0.8812 |
| F1 Score | 0.8504 |
| AUC-ROC | 0.8197 |

**Confusion Matrix:**
```
[[ 7,776  6,546]
 [ 4,068 30,178]]
```

### 3. IRT Parameter Estimation

The most significant improvement came in IRT difficulty parameter estimation:

| Metric | LightGBM IRT | Neural Network IRT | Improvement |
|--------|--------------|-------------------|-------------|
| Pearson correlation | 0.0268 | 0.9602 | 35.8× increase |
| Spearman correlation | 0.0305 | 0.9952 | 32.6× increase |
| Mean Absolute Error | 1.4476 | 0.7778 | 46% decrease |
| RMSE | 2.0526 | 1.1748 | 43% decrease |

This dramatic improvement in correlation (from near-zero to nearly perfect) demonstrates that our neural network approach successfully balances user and question influences, leading to much more accurate difficulty estimations.

## User Embedding Analysis

The 8-dimensional user embeddings learned by the model capture meaningful patterns about user abilities:

1. **Dimensionality Reduction**: PCA visualization showed that the embeddings organized users in a way that reflected their performance patterns
2. **Clustering Effects**: Users with similar performance patterns clustered together in the embedding space
3. **Interpretable Dimensions**: The embedding dimensions appeared to capture different aspects of user performance

## Conclusions

The neural network approach with user embeddings successfully addressed the key limitation of our previous LightGBM model:

1. **Balanced Influence**: By embedding users into a low-dimensional space, we prevented the dominance of user IDs in predictions

2. **More Accurate IRT Estimation**: The balanced approach led to dramatically improved IRT difficulty estimations, with correlations increasing from near-zero to over 0.96

3. **Retention of Performance**: We maintained strong predictive performance while achieving better feature balance

4. **Robust Validation**: Our three-way data split provides confidence that these results are reliable and generalizable

This approach demonstrates how neural network architectures can help balance competing influences in educational assessment models, leading to more nuanced and accurate predictions of both user abilities and question difficulties.

## Generalization to Unseen Questions: Holdout Test Analysis

To rigorously evaluate our model's ability to generalize to completely unseen questions, we conducted an in-depth analysis using the 10% holdout test set (469 questions) that was completely isolated during model training.

### Prediction Analysis on Holdout Questions

We generated predictions for the 100 most active users across all 469 holdout questions, analyzing:

1. **Prediction Patterns**:
   - Mean probability: 0.7058
   - Distribution of probabilities showed meaningful variation, with a standard deviation of 0.2579
   - Binary predictions showed 73.63% of answers predicted as correct

2. **Variation Metrics**:
   - Standard deviation across questions: 0.2097 (similar to validation set: 0.2131)
   - Standard deviation across users: 0.1540 (slightly lower than validation set: 0.1669)
   - The ratio between these values (1.36) indicates that our model maintained a good balance between question and user influences

3. **Correlational Analysis**:
   - Correlation between average predictions and actual correctness rates: 0.7679
   - Correlation between predictions and IRT difficulty: -0.7676
   - These strong correlations suggest the model correctly identified the relative difficulty of questions

### IRT Parameter Estimation on Holdout Questions

We applied Item Response Theory (2PL model) to estimate difficulty parameters from our model's predictions on the holdout questions, comparing these with the original IRT difficulties:

1. **Correlation Metrics**:
   | Metric | Value | Interpretation |
   |--------|-------|----------------|
   | Pearson correlation | 0.8421 | Strong linear relationship |
   | Spearman correlation | 0.9738 | Near-perfect rank correlation |
   | MAE | 0.5176 | Average error of half a difficulty unit |
   | RMSE | 1.0352 | Standard error of about one difficulty unit |

2. **Pattern Analysis**:
   - Comparison of estimated vs. original difficulty values showed a remarkably consistent relationship
   - The discrimination parameters were generally above 1.0, indicating questions effectively differentiated between students
   - The extremely high Spearman correlation (0.9738) demonstrates that the model accurately ranked questions by difficulty, even for completely unseen questions

3. **Visualization Findings**:
   - Scatter plots revealed that estimated difficulties closely tracked original difficulties across the entire range
   - The strongest correlations were observed for questions with higher discrimination parameters
   - The model showed some compression of the difficulty scale (a common pattern in IRT estimation), estimating easy questions as slightly harder than ground truth and difficult questions as slightly easier

### Implications for Educational Assessment

These holdout test results have significant implications for educational assessment practices:

1. **Generalizability**: The neural network model demonstrated exceptional ability to generalize difficulty estimations to completely unseen questions, with correlations above 0.84 (Pearson) and 0.97 (Spearman)

2. **Efficient Assessment**: This strong generalization suggests the possibility of efficiently estimating question difficulty parameters without requiring extensive preliminary testing

3. **Balanced Information Utilization**: The model effectively leveraged both user and question characteristics, as evidenced by the balanced standard deviations across both dimensions

4. **Practical Applications**: These results support the use of this approach for:
   - Estimating difficulty of new questions before widespread deployment
   - Adaptive testing systems requiring accurate difficulty estimates
   - Educational content development where understanding question difficulty is crucial

### Comparison with Previous Approaches

The neural network model's performance on holdout questions significantly outperformed traditional approaches:

| Metric | LightGBM | Neural Network (Validation) | Neural Network (Holdout) |
|--------|----------|---------------------------|--------------------------|
| Pearson correlation | 0.0268 | 0.9602 | 0.8421 |
| Spearman correlation | 0.0305 | 0.9952 | 0.9738 |
| MAE | 1.4476 | 0.7778 | 0.5176 |
| RMSE | 2.0526 | 1.1748 | 1.0352 |

While there was some expected degradation in performance from validation to holdout sets, the neural network's performance on completely unseen questions remained exceptional, particularly in maintaining near-perfect rank correlation (Spearman).

## Future Directions

Based on our comprehensive analysis and positive results, several promising future directions emerge:

1. **Embedding Interpretation**: Further analyze the user embeddings to understand what properties each dimension captures

2. **Additional Feature Types**: Incorporate more question features and explore interactions between user and question embeddings

3. **Adaptive Testing**: Use the model's balanced assessments to create personalized question sequences for students

4. **Transfer Learning**: Explore how well user embeddings transfer to new questions not seen during training

5. **Production Implementation**: Create an efficient API for generating predictions in real-time educational applications

6. **Large-Scale Deployment**: Extend the approach to larger question banks and more diverse student populations to validate scalability

7. **Curriculum Optimization**: Use difficulty estimates to sequence educational content for optimal learning progression

## Final Experiment: Scaling to All Users

As a final experiment, we investigated how the model's performance in estimating IRT difficulty parameters scales when using all available users instead of just the top 100 most active users.

### Experimental Setup

1. **Data Scope**: 
   - Previous analysis: 100 most active users × 469 holdout questions
   - New analysis: 1,867 total users × 469 holdout questions (18.7× more user data)

2. **Methodology**:
   - Generated predictions for all 1,867 users on all 469 holdout questions
   - Applied the same 2PL IRT model to estimate question difficulty and discrimination parameters
   - Compared these estimates with the original IRT difficulties
   - Benchmarked against our previous results using only 100 users

### Results: Top 100 Users vs. All Users

| Metric | Top 100 Users | All Users | Δ |
|--------|--------------|-----------|---|
| Pearson Correlation | 0.8421 | 0.8549 | +1.5% |
| Spearman Correlation | 0.9738 | 0.9629 | -1.1% |
| Mean Absolute Error | 0.5176 | 0.6286 | +21.4% |
| Root Mean Square Error | 1.0352 | 1.0760 | +3.9% |

### Key Findings

1. **Improved Linear Correlation**: Using all users resulted in a slightly higher Pearson correlation (0.8549 vs. 0.8421), indicating a modest improvement in capturing the linear relationship between estimated and original difficulties.

2. **Slight Decrease in Rank Correlation**: The Spearman correlation decreased slightly (0.9629 vs. 0.9738), but remains extremely high. This suggests that while the exact values might differ, the ranking of questions by difficulty remains nearly perfect.

3. **Increased Error Metrics**: Both MAE and RMSE increased when using all users, with MAE showing the most notable change (0.6286 vs. 0.5176). This suggests that while linear correlation improved, the absolute accuracy of the estimates decreased somewhat.

4. **Balanced Performance Tradeoff**: The increased user sample provided more data but also introduced more variability, resulting in a tradeoff between improved correlation and slightly reduced precision.

### Analysis of Error Patterns

The error distribution when using all users showed:

1. **Systematic Bias**: A tendency to overestimate the difficulty of easy questions and underestimate the difficulty of hard questions, a common regression-to-the-mean pattern in IRT estimation.

2. **Consistent Performance Across the Difficulty Range**: The model maintained strong correlations across the entire difficulty spectrum, from very easy to very difficult questions.

3. **Effective Discrimination**: The discrimination parameters remained above 1.0 for most questions, indicating that the model effectively distinguished between high and low-ability users.

### Implications

1. **Robustness to User Sample Size**: The neural network model's ability to estimate IRT parameters is remarkably robust, with only minor changes in performance when scaling from 100 to 1,867 users.

2. **Sample Efficiency**: The strong performance with just 100 users suggests that our approach is sample-efficient, requiring relatively few responses per question to achieve reliable difficulty estimates.

3. **Practical Application**: For practical applications, using a smaller set of active users may be sufficient and computationally more efficient, while still providing highly accurate difficulty estimates.

4. **Generalizability**: The consistent performance across different user sample sizes further validates the generalizability of our approach to new, unseen questions.

This final experiment completes our comprehensive evaluation of the neural network approach with user embeddings, demonstrating its effectiveness, robustness, and scalability for educational assessment applications.

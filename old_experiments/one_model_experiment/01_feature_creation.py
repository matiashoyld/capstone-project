import pandas as pd
import os
import re
import numpy as np

# Function to extract text features from question titles
def extract_text_features(text):
    """
    Extract advanced features from text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary of extracted features
    """
    if pd.isna(text) or not isinstance(text, str):
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'digit_count': 0,
            'special_char_count': 0,
            'mathematical_symbols': 0,
            'latex_expressions': 0
        }
    
    # Count words and characters
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    char_count = len(text)
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Count digits
    digit_count = sum(c.isdigit() for c in text)
    
    # Count special characters
    special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
    
    # Count mathematical symbols (including common symbols and operators)
    math_symbols = set(['+', '-', '*', '/', '=', '<', '>', '±', '≤', '≥', '≠', '≈', '∞', '∫', '∑', '∏', '√', '^', '÷', '×', '∆', '∇', '∂'])
    mathematical_symbols = sum(text.count(sym) for sym in math_symbols)
    
    # Count LaTeX expressions (approximate using pattern matching)
    latex_patterns = [r'\\\w+', r'\$.*?\$', r'\\\(.*?\\\)', r'\\\[.*?\\\]']
    latex_expressions = sum(len(re.findall(pattern, text)) for pattern in latex_patterns)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'mathematical_symbols': mathematical_symbols,
        'latex_expressions': latex_expressions
    }

def jaccard_similarity(str1, str2):
    """
    Calculate Jaccard similarity between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Jaccard similarity score (intersection over union)
    """
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0
    
    if len(str1) == 0 or len(str2) == 0:
        return 0
        
    # Create sets of words
    set1 = set(re.findall(r'\b\w+\b', str1.lower()))
    set2 = set(re.findall(r'\b\w+\b', str2.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0
    
    return intersection / union

def calculate_option_similarities_and_length(row):
    """
    Calculate Jaccard similarities between question options and average option length.
    
    Args:
        row: DataFrame row containing option_a through option_e
        
    Returns:
        Dictionary with jaccard_similarity_std and avg_option_length
    """
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    options = [str(row[col]) if pd.notna(row[col]) else "" for col in option_cols]
    valid_options = [opt for opt in options if opt]
    
    # Calculate average option length
    option_lengths = [len(opt) for opt in valid_options]
    avg_option_length = np.mean(option_lengths) if option_lengths else 0
    
    # Calculate average option word count
    option_word_counts = [len(re.findall(r'\b\w+\b', opt.lower())) for opt in valid_options]
    avg_option_word_count = np.mean(option_word_counts) if option_word_counts else 0
    
    # If less than 2 valid options, return zeros for similarity
    if len(valid_options) < 2:
        return {
            'jaccard_similarity_std': 0,
            'avg_option_length': avg_option_length,
            'avg_option_word_count': avg_option_word_count
        }
    
    # Calculate Jaccard similarities
    similarities = []
    for i in range(len(valid_options)):
        for j in range(i+1, len(valid_options)):
            sim = jaccard_similarity(valid_options[i], valid_options[j])
            similarities.append(sim)
    
    # Calculate standard deviation of similarities
    std_similarity = np.std(similarities) if len(similarities) > 1 else 0
    
    return {
        'jaccard_similarity_std': std_similarity,
        'avg_option_length': avg_option_length,
        'avg_option_word_count': avg_option_word_count
    }

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV files
master_translated = pd.read_csv(os.path.join(current_dir, 'data/master_translated.csv'))
questions_master = pd.read_csv(os.path.join(current_dir, 'data/questions_master.csv'))
question_difficulties_irt = pd.read_csv(os.path.join(current_dir, 'data/question_difficulties_irt.csv'))

# Columns to keep from each dataset
master_cols = ['answer_id', 'is_correct', 'user_id', 'grade_id', 'user_level', 'question_id']
irt_cols = ['question_id', 'irt_difficulty', 'original_difficulty']
question_cols = [
    'question_id', 'question_title', 
    'option_a', 'option_b', 'option_c', 'option_d', 'option_e',
    'correct_option_letter', 'avg_steps', 'skills', 'level', 
    'num_misconceptions', 'topic_id', 'topic_name', 'subject_id', 
    'subject_name', 'axis_id', 'axis_name',
    'count_a', 'count_b', 'count_c', 'count_d', 'count_e',
    'total_count', 'has_image'
]

# Apply text feature extraction to question_title before merging
print("Extracting text features from question titles...")
question_title_features = questions_master['question_title'].apply(extract_text_features)

# Convert the dictionary series to separate columns
text_feature_df = pd.DataFrame(question_title_features.tolist())

# Add prefix to avoid column name conflicts
text_feature_df = text_feature_df.add_prefix('title_')

# Calculate similarities between question options and average length
print("Calculating option similarities and lengths...")
option_features = questions_master.apply(calculate_option_similarities_and_length, axis=1)
option_features_df = pd.DataFrame(option_features.tolist())

# Concatenate the text features and option features with the questions_master dataframe
questions_master = pd.concat([questions_master, text_feature_df, option_features_df], axis=1)

# Add these columns to the list of columns to keep
text_feature_cols = [
    'title_word_count', 'title_char_count', 'title_avg_word_length',
    'title_digit_count', 'title_special_char_count', 
    'title_mathematical_symbols', 'title_latex_expressions'
]

option_feature_cols = [
    'jaccard_similarity_std', 'avg_option_length', 'avg_option_word_count'
]

question_cols.extend(text_feature_cols)
question_cols.extend(option_feature_cols)

# Select only the required columns from each dataframe
master_translated = master_translated[master_cols]
question_difficulties_irt = question_difficulties_irt[irt_cols]
questions_master = questions_master[question_cols]

# Print statistics about the extracted text features
print("\nText feature statistics for question titles:")
print(questions_master[text_feature_cols].describe())

# Print statistics about the option features
print("\nOption feature statistics:")
print(questions_master[option_feature_cols].describe())

# Merge the dataframes on question_id
merged_df = master_translated.merge(
    question_difficulties_irt, on='question_id', how='inner'
).merge(
    questions_master, on='question_id', how='inner'
)

# Print data info before filtering
print(f"Original merged dataframe shape: {merged_df.shape}")

# Filter out questions with less than 10 total_count
# and irt_difficulty less than -6
questions_to_keep = merged_df['question_id'].isin(
    merged_df[(merged_df['total_count'] >= 10) & 
              (merged_df['irt_difficulty'] >= -6)]['question_id'].unique()
)
filtered_df = merged_df[questions_to_keep]

# Print data info after filtering
print(f"Filtered dataframe shape: {filtered_df.shape}")
print(f"Removed {merged_df.shape[0] - filtered_df.shape[0]} rows due to filtering criteria")

# Count of unique questions before and after filtering
unique_questions_before = merged_df['question_id'].nunique()
unique_questions_after = filtered_df['question_id'].nunique()
print(f"Unique questions before filtering: {unique_questions_before}")
print(f"Unique questions after filtering: {unique_questions_after}")
print(f"Removed {unique_questions_before - unique_questions_after} unique questions")

# Display information about the filtered dataframe
print("\nFiltered dataframe columns:")
print(filtered_df.columns.tolist())
print("\nSample of filtered dataframe:")
print(filtered_df.head())

# Save the filtered dataframe to a new CSV file
output_path = os.path.join(current_dir, 'data/merged_features_filtered.csv')
filtered_df.to_csv(output_path, index=False)
print(f"\nFiltered dataframe saved to: {output_path}")

# Basic data quality check
print("\nNull values in each column:")
print(filtered_df.isnull().sum())

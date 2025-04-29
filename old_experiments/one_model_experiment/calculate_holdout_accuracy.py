import pandas as pd
import numpy as np

def main():
    # Load the binary prediction matrix and holdout questions
    binary = pd.read_csv('predictions/08_holdout_binary_prediction_matrix.csv', index_col=0)
    holdout = pd.read_csv('data/07_holdout_test_questions.csv')
    
    print(f"Binary matrix shape: {binary.shape}")
    print(f"Number of holdout questions: {len(holdout)}")
    
    # Convert question_id to string in holdout dataframe
    holdout['question_id'] = holdout['question_id'].astype(str)
    
    # Create a mapping from question_id to is_correct
    holdout_q_correct = holdout.set_index('question_id')['is_correct']
    
    print(f"First 5 question IDs in binary matrix: {list(binary.columns[:5])}")
    print(f"First 5 question IDs in holdout: {list(holdout_q_correct.index[:5])}")
    
    # Calculate accuracy for each user
    avg_accuracy = []
    
    for user_id, row in binary.iterrows():
        correct_count = 0
        total = 0
        
        for q_id, pred in row.items():
            q_id_str = str(q_id)
            if q_id_str in holdout_q_correct.index:
                actual = holdout_q_correct[q_id_str]
                correct = (int(pred) == int(actual))
                correct_count += int(correct)
                total += 1
        
        if total > 0:
            acc = correct_count / total
            avg_accuracy.append(acc)
    
    if not avg_accuracy:
        print("Error: No matching question IDs found between prediction matrix and holdout set")
        return
    
    # Print overall statistics
    print(f'Average accuracy across users: {np.mean(avg_accuracy):.4f}')
    print(f'Min accuracy: {np.min(avg_accuracy):.4f}')
    print(f'Max accuracy: {np.max(avg_accuracy):.4f}')
    print(f'Std dev of accuracy: {np.std(avg_accuracy):.4f}')
    
    # Calculate precision, recall, and F1 score
    all_predictions = []
    all_actuals = []
    
    for user_id, row in binary.iterrows():
        for q_id, pred in row.items():
            q_id_str = str(q_id)
            if q_id_str in holdout_q_correct.index:
                actual = holdout_q_correct[q_id_str]
                all_predictions.append(int(pred))
                all_actuals.append(int(actual))
    
    if not all_predictions:
        print("Error: No matching question IDs found for metrics calculation")
        return
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    # Calculate metrics
    tp = np.sum((all_predictions == 1) & (all_actuals == 1))
    fp = np.sum((all_predictions == 1) & (all_actuals == 0))
    tn = np.sum((all_predictions == 0) & (all_actuals == 0))
    fn = np.sum((all_predictions == 0) & (all_actuals == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nOverall metrics:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')
    
    # Compare to training metrics
    print("\nComparison with validation set metrics from training:")
    print("Validation set metrics (from training):")
    print("Accuracy: 0.7815")
    print("Precision: 0.8218")
    print("Recall: 0.8812")
    print("F1 Score: 0.8504")

if __name__ == "__main__":
    main() 
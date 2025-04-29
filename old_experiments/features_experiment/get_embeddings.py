# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

# %%
# Fix the path since we're already in the features_experiment directory
questions = pd.read_csv('questions_filtered.csv')

# %%
questions.info()


# %%
# Concatenate question_title with all options for each row
def concatenate_text(row):
    # No need for search_document prefix with all-MiniLM-L6-v2 model
    text = f"Question: {row['question_title']}\n"
    
    option_letters = ['A', 'B', 'C', 'D', 'E']
    option_columns = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']
    
    for letter, option in zip(option_letters, option_columns):
        if option in row and pd.notna(row[option]):
            text += f"{letter}) {str(row[option])}\n"
    
    return text

# Create a new column with concatenated text
questions['concatenated_text'] = questions.apply(concatenate_text, axis=1)

# Generate embeddings for concatenated text
print("Generating embeddings...")
text_embeddings = model.encode(questions['concatenated_text'].tolist())

# Create a DataFrame with question_id and embedding
questions_embeddings = pd.DataFrame({
    'question_id': questions['question_id'],
    'embedding': list(text_embeddings)
})

# Display a sample of the DataFrame
print("\nSample of questions_embeddings DataFrame:")
print(questions_embeddings.head())

# Save embeddings to a pickle file
print("Saving embeddings to pickle file...")
questions_embeddings.to_pickle('questions_embeddings.pkl')

print("Done! Saved to questions_embeddings.pkl")

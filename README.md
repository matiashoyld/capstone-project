# Student Response Simulation for Question Difficulty Estimation

## Project Metadata

- **Authors:** Matías Hoyl
- **Data used:** Zapien edtech platform data, including student-question interactions and IRT ability scores
- **Data collection period:** 2018-2022

## Project Goal

This study simulates student responses to math questions using Large Language Models (LLMs) to estimate question difficulty. The experiment analyzes **50 students** across multiple topics, focusing on their interactions with **30 carefully selected questions**. Using the **Gemini Flash 2.0 Thinking** model, we simulate how students with different ability levels would respond to questions, taking into account their:

- Topic-specific IRT scores
- Age
- Previous performance patterns

The goal is to understand if LLM-simulated responses can accurately predict question difficulty and match actual student performance patterns.

## Methodology and Workflow

The experiment is structured in several steps, using **Python** for data manipulation, LLM interactions, and analysis. The workflow is contained in a Jupyter notebook with detailed explanations for replication. All manipulated datasets are stored in the project directory.

### Step 1: Data Preparation and Setup

1. Import required libraries and set up environment variables
2. Load the raw master dataset containing student-question interactions
3. Configure the Gemini LLM API for response simulation

### Step 2: Student Dataset Creation

1. Sort data chronologically by `answer_id`
2. Group by `user_id` and `topic_id` to aggregate statistics
3. Create a matrix of user topic levels for the top 50 students
4. Add student age information from grade data

### Step 3: Question Dataset Creation

1. Extract question metadata and response counts
2. Focus on questions from topic 452 (the topic with the most questions answered)
3. Select top 30 questions based on response count
4. Extract and map multiple-choice options to standardized format

### Step 4: LLM Response Simulation

1. Create question prompt templates incorporating student profiles
2. Define helper functions for response aggregation
3. Implement majority voting for multiple LLM responses
4. Run the experiment iterating through questions and users

### Step 5: Results Analysis

1. Record simulated responses and actual correct answers
2. Calculate accuracy metrics
3. Save results for further analysis

## Key Files

The experiment relies on the following data files:

| File Name | Description |
|-----------|-------------|
| `master.csv` | Main dataset with student-question interactions |
| `updated_user_profiles.csv` | Processed student data with topic levels |
| `top_30_questions.csv` | Selected questions for the experiment |
| `experiment_results.csv` | Final simulation results |

## Reproducibility

All code includes detailed comments and explanations to ensure reproducibility. The notebook provides a complete workflow from raw data processing to final analysis.

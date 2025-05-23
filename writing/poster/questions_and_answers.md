# Poster questions and answers

## What is this about?

This research focuses on predicting the difficulty of math questions more efficiently. We go beyond traditional features by using Large Language Models (LLMs) to extract pedagogical insights from each question. We then train a neural network to simulate how the students from the training data would likely answer new, unseen questions.

## Why should I care?

Traditional difficulty estimation requires slow, expensive student pre-testing, creating bottlenecks for teachers and assessment developers. Our approach provides accurate estimates without any student testing, speeding up development, reducing costs, and potentially enhancing adaptive learning systems that personalize student experiences.

## How did you do it?

1. Feature Extraction: Gathered traditional data about each question (like word count) and used LLMs to identify deeper pedagogical characteristics (see table).
2. Correctness Prediction: Trained a neural network using 90% of our data (which included real student-question interactions). This network learned to predict if a student would answer a question correctly. We then used it to predict outcomes for the remaining 10% of unseen questions.
3. Difficulty Estimation: Using these simulated student answers for the unseen questions, we calculated their difficulty scores (using a statistical method called 1PL IRT).
4. Evaluation: Compared our estimated difficulties with actual student-derived difficulty parameters, achieving a 0.78 correlation on unseen questions

# What features did the LLM produce?

- Solution Steps: How many steps does it take to solve?
- Cognitive Level: What type of thinking is required (e.g., Apply, Analyze)?
- Misconceptions: How many common student mistakes does this question entail?
- Expression Complexity: How deeply are expressions nested?
- Information Gap: How much unstated knowledge is required?
- Distractor Plausibility: Are wrong answers tempting and realistic?
- Problem Archetype: What type of problem is it (e.g., word problem, equation solving, etc.)?
- Knowledge Dimension: Does this problem test factual, conceptual, or procedural knowledge?
- Real-World Context: Is the problem abstract or situated in a real scenario?

## Where does the data come from?

From Zapien, an adaptive math learning platform in Chile. It includes 251,851 student responses to 4,696 unique math questions, provided by 1,875 different students.

## How efficient is your approach?

We compared our model's accuracy to traditional methods that use varying amounts of real student data. Our approach (using no real student testing on new questions) achieved the same accuracy as traditional methods would achieve using about 5,800 real student responses (about 23% of typical testing data)
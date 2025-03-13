# %%
import pandas as pd
import numpy as np
from scipy import optimize
import time
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm  # For progress bars
import multiprocessing
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pytorch for GPU acceleration
try:
    import torch
    HAS_TORCH = True
    # Check if MPS (Apple Silicon GPU) is available
    HAS_MPS = torch.backends.mps.is_available()
    if HAS_MPS:
        DEVICE = torch.device("mps")
        logger.info("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        logger.info("Using NVIDIA GPU acceleration (CUDA)")
    else:
        DEVICE = torch.device("cpu")
        logger.info("GPU not available, using CPU")
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    DEVICE = None
    logger.info("PyTorch not installed, using NumPy on CPU")

# Determine number of CPU cores to use
NUM_CORES = min(multiprocessing.cpu_count(), 8)  # Use at most 8 cores
logger.info(f"Using {NUM_CORES} CPU cores for parallelization")

# Start timing
start_time = time.time()
logger.info("Starting IRT estimation process")

answers = pd.read_csv('../data/new/master_translated.csv')

# %%
# Display information about the dataframe
answers.info()

# %%
# Filter to keep only questions with 10 or more answers
question_counts = answers['question_id'].value_counts()
questions_to_keep = question_counts[question_counts >= 10].index
filtered_answers = answers[answers['question_id'].isin(questions_to_keep)]

logger.info(f"Original number of questions: {answers['question_id'].nunique()}")
logger.info(f"Number of questions with 10+ answers: {filtered_answers['question_id'].nunique()}")
logger.info(f"Original number of answers: {len(answers)}")
logger.info(f"Number of answers after filtering: {len(filtered_answers)}")

# %%
# Use either the full IRT model or the simplified approach
USE_FULL_IRT = True

# IRT Model type: 1PL or 2PL
# 1PL (Rasch): P(correct) = 1 / (1 + exp(-(ability - difficulty)))
# 2PL: P(correct) = 1 / (1 + exp(-discrimination * (ability - difficulty)))
IRT_MODEL = "2PL"  # Changed from 1PL to 2PL
logger.info(f"Using {IRT_MODEL} model with {'GPU acceleration' if HAS_TORCH and HAS_MPS else 'CPU parallelization'}")

# Enable GPU acceleration if available
USE_GPU = HAS_TORCH and HAS_MPS

# To use the most efficient implementation
if USE_FULL_IRT:
    logger.info("Using full IRT model - this may take a long time")
    
    # Create a sparse matrix for IRT analysis
    # Each row represents a student (user_id), each column a question (question_id)
    # Values are 1 for correct answers, 0 for incorrect
    
    # First create mappings from IDs to indices
    user_ids = filtered_answers['user_id'].unique()
    question_ids = filtered_answers['question_id'].unique()
    
    n_students = len(user_ids)
    n_questions = len(question_ids)
    
    logger.info(f"Creating response matrix for {n_students} users and {n_questions} questions")
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    question_id_to_idx = {question_id: idx for idx, question_id in enumerate(question_ids)}
    
    # Initialize response matrix with NaN (for missing values)
    response_matrix = np.full((n_students, n_questions), np.nan)
    
    # Fill in the response matrix
    logger.info("Filling response matrix - this may take a few minutes")
    
    # Use faster NumPy vectorized approach for filling the matrix
    # Create arrays for each column
    user_indices = np.array([user_id_to_idx[uid] for uid in filtered_answers['user_id']])
    question_indices = np.array([question_id_to_idx[qid] for qid in filtered_answers['question_id']])
    is_correct_values = filtered_answers['is_correct'].astype(float).values
    
    # Fill the matrix in one operation
    response_matrix[user_indices, question_indices] = is_correct_values
    logger.info("Response matrix filled successfully")
    
    # Create nan_mask here since we'll use it multiple times
    nan_mask = np.isnan(response_matrix)
    
    if USE_GPU and HAS_TORCH:
        # PyTorch implementation of 2PL IRT model for GPU acceleration
        logger.info(f"Using PyTorch GPU implementation for {IRT_MODEL} model")
        
        # Convert NumPy arrays to PyTorch tensors
        def to_tensor(array):
            return torch.tensor(array, dtype=torch.float32, device=DEVICE)
        
        # Convert response matrix to a tensor, replacing NaN with -1
        response_tensor = torch.tensor(np.nan_to_num(response_matrix, nan=-1), dtype=torch.float32, device=DEVICE)
        # Create a mask for valid (non-NaN) entries
        valid_mask = torch.tensor(~nan_mask, dtype=torch.bool, device=DEVICE)
        
        class IRT2PLModel(torch.nn.Module):
            def __init__(self, n_students, n_questions):
                super().__init__()
                # Initialize abilities with mean 0
                self.abilities = torch.nn.Parameter(torch.zeros(n_students, device=DEVICE))
                
                # Initialize difficulties based on proportion correct
                correct_proportions = np.nanmean(response_matrix, axis=0)
                correct_proportions = np.clip(correct_proportions, 0.01, 0.99)
                initial_difficulties = -np.log(correct_proportions / (1 - correct_proportions))
                self.difficulties = torch.nn.Parameter(torch.tensor(
                    initial_difficulties, dtype=torch.float32, device=DEVICE))
                
                # Initialize discrimination parameters to 1.0 (2PL model)
                # In 2PL, discrimination parameter controls how well a question differentiates
                # between students of different abilities
                self.discriminations = torch.nn.Parameter(torch.ones(n_questions, device=DEVICE))
                
                # Add a soft constraint to keep discriminations positive but not too large
                self.discr_min = 0.25
                self.discr_max = 4.0
            
            def forward(self):
                # Reshape for broadcasting
                abilities = self.abilities.view(-1, 1)
                difficulties = self.difficulties.view(1, -1)
                
                # For 2PL, we multiply (ability - difficulty) by discrimination
                # Reshape discrimination for broadcasting
                discriminations = self.discriminations.view(1, -1)
                
                # Calculate logits with discrimination factor
                logits = discriminations * (abilities - difficulties)
                
                # Convert to probabilities
                probs = torch.sigmoid(logits)
                
                return probs
            
            def log_likelihood(self):
                # Constrain discriminations to reasonable bounds using sigmoid
                # This is a soft constraint through penalty
                constrained_discr = torch.sigmoid(self.discriminations) * (self.discr_max - self.discr_min) + self.discr_min
                
                # Calculate probabilities
                probs = self.forward()
                
                # Get only valid entries
                valid_probs = probs[valid_mask]
                valid_responses = response_tensor[valid_mask]
                
                # Calculate log likelihood
                log_like = torch.sum(
                    valid_responses * torch.log(valid_probs + 1e-10) + 
                    (1 - valid_responses) * torch.log(1 - valid_probs + 1e-10)
                )
                
                # Add constraint to keep mean ability at 0
                ability_constraint = torch.mean(self.abilities) ** 2
                
                # Add penalty to prevent extreme discrimination values
                discr_constraint = torch.sum((self.discriminations < self.discr_min).float() * (self.discr_min - self.discriminations) ** 2 +
                                           (self.discriminations > self.discr_max).float() * (self.discriminations - self.discr_max) ** 2)
                
                # Return negative log likelihood for minimization
                return -log_like + 1000 * ability_constraint + 10 * discr_constraint
        
        def estimate_parameters_torch():
            # Create model
            model = IRT2PLModel(n_students, n_questions).to(DEVICE)
            
            # Use Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Training loop
            n_epochs = 1000
            best_loss = float('inf')
            best_params = None
            patience_counter = 0
            max_patience = 30
            
            # For saving checkpoints
            last_save_time = time.time()
            
            # Add convergence threshold - stop if improvement is very small
            convergence_threshold = 0.0001  # 0.01% improvement
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = model.log_likelihood()
                loss.backward()
                optimizer.step()
                
                # Adjust learning rate
                scheduler.step(loss.item())
                
                # Log progress more frequently in later epochs
                should_log = epoch % 10 == 0 if epoch < 300 else epoch % 5 == 0
                
                if should_log:
                    current_loss = loss.item()
                    logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {current_loss:.4f}")
                    
                    # Check for improvement
                    relative_improvement = 0
                    if best_loss != float('inf'):
                        relative_improvement = (best_loss - current_loss) / best_loss
                        
                    if current_loss < best_loss * 0.9999:  # Improvement threshold
                        best_loss = current_loss
                        best_params = (
                            model.abilities.detach().cpu().numpy(),
                            model.difficulties.detach().cpu().numpy(),
                            model.discriminations.detach().cpu().numpy()
                        )
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Log relative improvement
                    if epoch > 0 and epoch % 50 == 0:
                        logger.info(f"Relative improvement: {relative_improvement:.6f}")
                    
                    # Check convergence
                    if epoch > 100 and relative_improvement < convergence_threshold:
                        logger.info(f"Converged at epoch {epoch} with improvement {relative_improvement:.6f} < {convergence_threshold}")
                        if relative_improvement > 0:  # Still some improvement
                            logger.info("Continuing for fine-tuning")
                        
                    # Save checkpoint if 5 minutes have passed
                    current_time = time.time()
                    if current_time - last_save_time > 300:
                        # Save intermediate results
                        abilities = model.abilities.detach().cpu().numpy()
                        difficulties = model.difficulties.detach().cpu().numpy()
                        discriminations = model.discriminations.detach().cpu().numpy()
                        np.savez(
                            f'irt_torch_epoch{epoch}.npz',
                            abilities=abilities,
                            difficulties=difficulties,
                            discriminations=discriminations,
                            epoch=epoch
                        )
                        logger.info(f"Saved intermediate results at epoch {epoch}")
                        last_save_time = current_time
                
                # Early stopping check
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Return best parameters
            return best_params
        
        try:
            logger.info("Starting PyTorch optimization")
            result = estimate_parameters_torch()
            if len(result) == 3:  # 2PL model
                abilities, difficulties, discriminations = result
            else:  # 1PL model (fallback)
                abilities, difficulties = result
                discriminations = np.ones(n_questions)  # Default discrimination = 1
                
            logger.info("PyTorch optimization completed successfully")
            
            # Save the full results
            np.savez('irt_torch_results.npz', 
                    abilities=abilities, 
                    difficulties=difficulties,
                    discriminations=discriminations,
                    question_ids=question_ids,
                    user_ids=user_ids)
            logger.info("Saved PyTorch results to irt_torch_results.npz")
            
        except Exception as e:
            logger.error(f"Error in PyTorch IRT estimation: {str(e)}")
            logger.info("Falling back to CPU implementation")
            USE_GPU = False
    
    if not USE_GPU:
        # NumPy CPU implementation with parallelization
        logger.info(f"Using NumPy CPU implementation with parallelization for {IRT_MODEL} model")
        
        from joblib import Parallel, delayed
        
        # Helper function for calculating log-likelihoods in parallel
        def calculate_student_log_likelihood(student_idx, responses, probs):
            student_responses = responses[student_idx, :]
            student_probs = probs[student_idx, :]
            
            # Only use non-NaN values
            mask = ~np.isnan(student_responses)
            
            if np.sum(mask) == 0:
                return 0.0
            
            log_like = np.sum(
                student_responses[mask] * np.log(student_probs[mask]) + 
                (1 - student_responses[mask]) * np.log(1 - student_probs[mask])
            )
            
            return log_like
        
        def negative_log_likelihood_parallel(params, responses, nan_mask):
            """Parallelized negative log likelihood calculation for 2PL model"""
            n_students, n_questions = responses.shape
            
            # Split params into ability, difficulty, and discrimination parameters
            if IRT_MODEL == "2PL":
                # First n_students parameters are abilities
                abilities = params[:n_students]
                # Next n_questions parameters are discriminations
                discriminations = params[n_students:n_students + n_questions]
                # Last n_questions parameters are difficulties
                difficulties = params[n_students + n_questions:]
            else:  # 1PL
                # First n_students parameters are abilities
                abilities = params[:n_students]
                # Last n_questions parameters are difficulties
                difficulties = params[n_students:]
                # Set all discriminations to 1
                discriminations = np.ones(n_questions)
            
            # Calculate probabilities for each student-question pair
            abilities_2d = abilities.reshape(-1, 1)  # Convert to column vector
            difficulties_2d = difficulties.reshape(1, -1)  # Convert to row vector
            discriminations_2d = discriminations.reshape(1, -1)  # Convert to row vector
            
            # Calculate log-odds using broadcasting
            if IRT_MODEL == "2PL":
                logits = discriminations_2d * (abilities_2d - difficulties_2d)
            else:  # 1PL
                logits = abilities_2d - difficulties_2d
            
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            # Clip probabilities to avoid numerical issues
            probs = np.clip(probs, 1e-10, 1-1e-10)
            
            # Calculate log likelihood for each student in parallel
            log_likes = Parallel(n_jobs=NUM_CORES)(
                delayed(calculate_student_log_likelihood)(i, responses, probs)
                for i in range(n_students)
            )
            
            # Sum the log likelihoods
            total_log_like = np.sum(log_likes)
            
            # Add penalty for discrimination values outside reasonable range
            if IRT_MODEL == "2PL":
                discr_min = 0.25
                discr_max = 4.0
                penalty = np.sum(
                    (discriminations < discr_min) * (discr_min - discriminations) ** 2 +
                    (discriminations > discr_max) * (discriminations - discr_max) ** 2
                ) * 10
                total_log_like -= penalty
            
            # Return negative log likelihood (for minimization)
            return -total_log_like
        
        def estimate_irt_parameters(response_matrix):
            logger.info(f"Starting parameter estimation for {n_students} students and {n_questions} questions")
            
            # Initialize parameters (abilities and difficulties)
            # Set initial abilities to 0 and difficulties based on proportion of correct answers
            initial_abilities = np.zeros(n_students)
            
            # Calculate initial difficulties based on proportion of correct answers
            # Higher proportion of correct answers means lower difficulty
            correct_proportions = np.nanmean(response_matrix, axis=0)
            # Handle extreme proportions
            correct_proportions = np.clip(correct_proportions, 0.01, 0.99)
            # Convert to logits (log-odds)
            initial_difficulties = -np.log(correct_proportions / (1 - correct_proportions))
            
            if IRT_MODEL == "2PL":
                # Initialize discrimination parameters to 1.0
                initial_discriminations = np.ones(n_questions)
                # Constraint: mean ability should be 0 for identifiability
                initial_params = np.concatenate([initial_abilities, initial_discriminations, initial_difficulties])
            else:  # 1PL
                # Constraint: mean ability should be 0 for identifiability
                initial_params = np.concatenate([initial_abilities, initial_difficulties])
            
            # Define constraint that mean ability is 0
            def ability_constraint(params):
                abilities = params[:n_students]
                return np.mean(abilities)
            
            constraints = {'type': 'eq', 'fun': ability_constraint}
            
            # Set up callback for progress reporting
            iteration = [0]
            last_time = [time.time()]
            
            def callback(xk):
                iteration[0] += 1
                current_time = time.time()
                # Only log every 10 iterations or if at least 5 minutes have passed
                if iteration[0] % 10 == 0 or (current_time - last_time[0] > 300):
                    logger.info(f"Optimization iteration {iteration[0]} (elapsed: {current_time - start_time:.1f} sec)")
                    last_time[0] = current_time
                    
                # Save intermediate results every 50 iterations
                if iteration[0] % 50 == 0:
                    temp_abilities = xk[:n_students]
                    if IRT_MODEL == "2PL":
                        temp_discriminations = xk[n_students:n_students + n_questions]
                        temp_difficulties = xk[n_students + n_questions:]
                        np.savez(
                            f'irt_intermediate_iter{iteration[0]}.npz',
                            abilities=temp_abilities,
                            discriminations=temp_discriminations,
                            difficulties=temp_difficulties,
                            iteration=iteration[0]
                        )
                    else:  # 1PL
                        temp_difficulties = xk[n_students:]
                        np.savez(
                            f'irt_intermediate_iter{iteration[0]}.npz',
                            abilities=temp_abilities,
                            difficulties=temp_difficulties,
                            iteration=iteration[0]
                        )
                    logger.info(f"Saved intermediate results at iteration {iteration[0]}")
                return False
            
            # Optimize parameters
            logger.info("Starting optimization - this will take a while")
            result = optimize.minimize(
                fun=negative_log_likelihood_parallel,
                x0=initial_params,
                args=(response_matrix, nan_mask),
                method='SLSQP',
                constraints=constraints,
                callback=callback,
                options={'disp': True, 'maxiter': 300, 'ftol': 1e-4}
            )
            
            logger.info(f"Optimization completed in {iteration[0]} iterations")
            logger.info(f"Optimization success: {result.success}, message: {result.message}")
            
            # Extract optimized parameters
            optimized_params = result.x
            abilities = optimized_params[:n_students]
            
            if IRT_MODEL == "2PL":
                discriminations = optimized_params[n_students:n_students + n_questions]
                difficulties = optimized_params[n_students + n_questions:]
                return abilities, difficulties, discriminations
            else:  # 1PL
                difficulties = optimized_params[n_students:]
                return abilities, difficulties
        
        # Run the parameter estimation
        logger.info(f"Estimating {IRT_MODEL} IRT parameters with parallelization (this may take hours)...")
        try:
            result = estimate_irt_parameters(response_matrix)
            if len(result) == 3:  # 2PL model
                abilities, difficulties, discriminations = result
            else:  # 1PL model (fallback)
                abilities, difficulties = result
                discriminations = np.ones(n_questions)  # Default discrimination = 1
                
            # Save the full results immediately
            np.savez('irt_full_results.npz', 
                    abilities=abilities, 
                    difficulties=difficulties,
                    discriminations=discriminations,
                    question_ids=question_ids,
                    user_ids=user_ids)
            logger.info("Saved full results to irt_full_results.npz")
        except Exception as e:
            logger.error(f"Error in IRT estimation: {str(e)}")
            # Try to load most recent intermediate results
            import glob
            intermediate_files = glob.glob('irt_intermediate_iter*.npz')
            if intermediate_files:
                # Find latest intermediate file
                latest_file = max(intermediate_files, key=lambda x: int(x.split('iter')[1].split('.')[0]))
                logger.info(f"Loading latest intermediate results from {latest_file}")
                data = np.load(latest_file)
                abilities = data['abilities']
                difficulties = data['difficulties']
                if 'discriminations' in data:
                    discriminations = data['discriminations']
                else:
                    discriminations = np.ones(n_questions)
                logger.info(f"Loaded parameters from iteration {data['iteration']}")
            else:
                logger.error("No intermediate results found. Falling back to simplified approach.")
                # Fall back to simplified approach
                USE_FULL_IRT = False
                discriminations = np.ones(n_questions)  # Default discrimination = 1

# OPTION 2: SIMPLIFIED IRT MODEL (much faster but still reasonable)
if not USE_FULL_IRT:
    logger.info("Using simplified IRT estimation (faster approach)")
    # Instead of joint estimation of all parameters, we'll use a simpler approach
    # based on the observed proportion of correct answers
    
    # First, calculate the proportion of correct answers for each question
    question_stats = filtered_answers.groupby('question_id')['is_correct'].agg(['count', 'mean'])
    question_stats.columns = ['num_answers', 'proportion_correct']
    
    # Calculate difficulty using the logit transformation
    # difficulty = -log(p/(1-p)) where p is the proportion of correct answers
    # Clip to avoid extreme values
    question_stats['proportion_correct'] = question_stats['proportion_correct'].clip(0.01, 0.99)
    question_stats['irt_difficulty'] = -np.log(question_stats['proportion_correct'] / (1 - question_stats['proportion_correct']))
    
    logger.info(f"Calculated simplified IRT difficulties for {len(question_stats)} questions")
    
    # Store question IDs and difficulties for later use
    question_ids = question_stats.index.values
    difficulties = question_stats['irt_difficulty'].values
    
    # For simplified 2PL, we just set all discriminations to 1
    discriminations = np.ones(len(question_ids))
    
    # We don't calculate student abilities in the simplified model
    abilities = None

# %%
# Create a dataframe with question difficulties
logger.info("Creating results dataframe")
difficulty_df = pd.DataFrame({
    'question_id': question_ids,
    'irt_difficulty': difficulties,
    'original_difficulty': [answers[answers['question_id'] == qid]['difficulty'].iloc[0] for qid in question_ids]
})

# For 2PL model, add discrimination parameters
if IRT_MODEL == "2PL" and 'discriminations' in locals():
    difficulty_df['discrimination'] = discriminations
    # Sort by difficulty and discrimination
    difficulty_df = difficulty_df.sort_values(['irt_difficulty', 'discrimination'])
else:
    # Sort by IRT difficulty
    difficulty_df = difficulty_df.sort_values('irt_difficulty')

# Display the first few questions with their difficulties
logger.info("Top 10 easiest questions:")
print(difficulty_df.head(10))

logger.info("\nTop 10 hardest questions:")
print(difficulty_df.tail(10))

# If 2PL, also show questions with highest and lowest discrimination
if IRT_MODEL == "2PL" and 'discriminations' in locals():
    logger.info("\nTop 10 most discriminating questions:")
    print(difficulty_df.sort_values('discrimination', ascending=False).head(10))
    
    logger.info("\nTop 10 least discriminating questions:")
    print(difficulty_df.sort_values('discrimination').head(10))

# %%
# Analyze correlation between original and IRT difficulty
correlation = difficulty_df['original_difficulty'].corr(difficulty_df['irt_difficulty'])
logger.info(f"\nCorrelation between original and IRT difficulty: {correlation:.3f}")

# If 2PL, also analyze correlation between difficulty and discrimination
if IRT_MODEL == "2PL" and 'discriminations' in locals():
    discr_diff_corr = difficulty_df['irt_difficulty'].corr(difficulty_df['discrimination'])
    logger.info(f"Correlation between difficulty and discrimination: {discr_diff_corr:.3f}")

# %%
# If we ran the full model, also create a student ability dataframe
if USE_FULL_IRT and abilities is not None:
    ability_df = pd.DataFrame({
        'user_id': user_ids,
        'ability': abilities
    })
    
    # Sort by ability
    ability_df = ability_df.sort_values('ability')
    
    # Display distribution of abilities
    plt.figure(figsize=(10, 6))
    plt.hist(abilities, bins=50)
    plt.xlabel('Student Ability (θ)')
    plt.ylabel('Count')
    plt.title('Distribution of Student Abilities')
    plt.grid(True, alpha=0.3)
    plt.savefig('student_abilities.png')
    logger.info("Created student ability histogram")
    
    # Save ability results
    ability_df.to_csv('student_abilities_irt.csv', index=False)
    logger.info("Saved student abilities to 'student_abilities_irt.csv'")

# %%
# Plot original vs IRT difficulty
plt.figure(figsize=(10, 6))
plt.scatter(difficulty_df['original_difficulty'], difficulty_df['irt_difficulty'], alpha=0.5)
plt.xlabel('Original Difficulty')
plt.ylabel('IRT Difficulty (2PL)' if IRT_MODEL == "2PL" else 'IRT Difficulty (1PL)')
plt.title('Comparison of Original vs IRT Difficulty Parameters')
plt.grid(True, alpha=0.3)
plt.savefig('difficulty_comparison.png')

# For 2PL, also plot discrimination values
if IRT_MODEL == "2PL" and 'discriminations' in locals():
    plt.figure(figsize=(10, 6))
    plt.scatter(difficulty_df['irt_difficulty'], difficulty_df['discrimination'], alpha=0.5)
    plt.xlabel('IRT Difficulty')
    plt.ylabel('Discrimination Parameter')
    plt.title('Difficulty vs Discrimination Parameters')
    plt.grid(True, alpha=0.3)
    plt.savefig('difficulty_discrimination.png')
    logger.info("Created difficulty vs discrimination plot")

# %%
# Save the results
difficulty_df.to_csv('question_difficulties_irt.csv', index=False)
logger.info("Results saved to 'question_difficulties_irt.csv'")

# Log total execution time
end_time = time.time()
logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

# %%
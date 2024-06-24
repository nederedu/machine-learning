import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load the dataset
data = pd.read_csv('insurance.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data['region'] = label_encoder.fit_transform(data['region'])

# Define the risk function
def risk_function(state):
    # Convert state values to actual variable values
    age = state[0] * (65 - 18) / 9 + 18  # Age range: 18-65
    bmi = state[1] * (40 - 15) / 9 + 15  # BMI range: 15-40
    smoker = state[2]  # Smoker: 0 or 1
    region = state[3]  # Region: 0-3
    
    return age * 0.2 + bmi * 0.3 + smoker * 2.0 + region * 0.1

# Normalize the risk function for optimization (minimization problem)
def normalized_risk_function(state):
    return -risk_function(state)

# Define the fitness function
fitness = mlrose.CustomFitness(normalized_risk_function)

# Helper function to run an algorithm and collect metrics with convergence criteria
def run_algorithm(problem, algorithm, max_attempts=100, *args, **kwargs):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = algorithm(problem, *args, curve=True, max_attempts=max_attempts, random_state=SEED, **kwargs)
    end_time = time.time()
    wall_clock_time = end_time - start_time
    evaluations = len(fitness_curve)
    return best_state, best_fitness, fitness_curve, evaluations, wall_clock_time

# Function to run experiments for a fixed problem size
def run_experiment(size, algorithm, *args, **kwargs):
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=10)
    best_state, best_fitness, fitness_curve, evaluations, wall_clock_time = run_algorithm(
        problem, algorithm, *args, **kwargs)
    return {
        'best_state': best_state,
        'best_fitness': -best_fitness,
        'fitness_curve': fitness_curve,
        'evaluations': evaluations,
        'wall_clock_time': wall_clock_time
    }

# Define the fixed problem size
problem_size = 4  # Size is 4, without the number of children

# Run experiments for each algorithm with convergence criteria
result_rhc = run_experiment(problem_size, mlrose.random_hill_climb, max_iters=1000, restarts=10, max_attempts=100)
result_sa = run_experiment(problem_size, mlrose.simulated_annealing, schedule=mlrose.ExpDecay(), max_iters=1000, max_attempts=100)
result_ga = run_experiment(problem_size, mlrose.genetic_alg, pop_size=200, mutation_prob=0.1, max_iters=1000, max_attempts=100)
result_mimic = run_experiment(problem_size, mlrose.mimic, pop_size=200, keep_pct=0.2, max_iters=1000, max_attempts=100)

# Find the common axis limits
all_fitness_curves = [result_rhc['fitness_curve'], result_sa['fitness_curve'], result_ga['fitness_curve'], result_mimic['fitness_curve']]
y_min = min(curve.min() for curve in all_fitness_curves)
y_max = max(curve.max() for curve in all_fitness_curves)
x_max = max(len(curve) for curve in all_fitness_curves)

# Plotting function for the fitness curves separately with the same scale
def plot_fitness_curve(result, algorithm_name, y_min, y_max, x_max):
    plt.figure(figsize=(10, 5))
    fitness_curve = result['fitness_curve']
    plt.plot(fitness_curve)
    plt.ylim(y_min, y_max)
    plt.xlim(0, x_max)
    plt.title(f'Fitness by Iterations - {algorithm_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()

# Generate separate plots for each algorithm with the same scale
plot_fitness_curve(result_rhc, 'Randomized Hill Climbing', y_min, y_max, x_max)
plot_fitness_curve(result_sa, 'Simulated Annealing', y_min, y_max, x_max)
plot_fitness_curve(result_ga, 'Genetic Algorithm', y_min, y_max, x_max)
plot_fitness_curve(result_mimic, 'MIMIC', y_min, y_max, x_max)

# Print detailed results for each algorithm
def print_detailed_results(result, algorithm_name):
    best_state = result['best_state']
    best_state_mapped = [
        best_state[0] * (65 - 18) / 9 + 18,
        best_state[1] * (40 - 15) / 9 + 15,
        best_state[2],
        best_state[3]
    ]
    print(f"{algorithm_name}:")
    print("Best state (encoded):", best_state)
    print("Best state (mapped):", best_state_mapped)
    print("Best fitness:", result['best_fitness'])
    print("Function Evaluations:", result['evaluations'])
    print("Wall Clock Time (s):", result['wall_clock_time'])
    print("-" * 40)

print_detailed_results(result_rhc, 'Randomized Hill Climbing')
print_detailed_results(result_sa, 'Simulated Annealing')
print_detailed_results(result_ga, 'Genetic Algorithm')
print_detailed_results(result_mimic, 'MIMIC')

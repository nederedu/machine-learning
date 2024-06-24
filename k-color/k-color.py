import pandas as pd
import mlrose_hiive as mlrose
import networkx as nx
import matplotlib.pyplot as plt
import time

# Load the dataset
file_path = 'insurance.csv'  # Update with your local file path
insurance_data = pd.read_csv(file_path)

# Reduce the dataset size for quicker testing
insurance_data = insurance_data.sample(n=100, random_state=42).reset_index(drop=True)

# Construct the graph based on billed charges similarity (within $5000)
G_charges = nx.Graph()
charges_threshold = 3000

# Add nodes with attributes
for index, row in insurance_data.iterrows():
    G_charges.add_node(index, charges=row['charges'])

# Add edges based on billed charges similarity
for i in range(len(insurance_data)):
    for j in range(i + 1, len(insurance_data)):
        if abs(insurance_data.at[i, 'charges'] - insurance_data.at[j, 'charges']) <= charges_threshold:
            G_charges.add_edge(i, j)

# Define the fitness function
edges = list(G_charges.edges)
fitness = mlrose.MaxKColor(edges)

# Set up the optimization problem
problem = mlrose.DiscreteOpt(length=len(insurance_data), fitness_fn=fitness, maximize=False, max_val=10)

# Function to run an algorithm and collect data
def run_algorithm(algorithm, problem, **kwargs):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = algorithm(problem, curve=True, **kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    function_evaluations = len(fitness_curve)
    
    return best_state, best_fitness, fitness_curve, function_evaluations, run_time

# Run algorithms
results = {}

# Randomized Hill Climbing
rhc_params = {'max_iters': 500, 'random_state': 42}
best_state_rhc, best_fitness_rhc, fitness_curve_rhc, evals_rhc, time_rhc = run_algorithm(
    mlrose.random_hill_climb, problem, **rhc_params)
results['Randomized Hill Climbing'] = (best_state_rhc, best_fitness_rhc, fitness_curve_rhc, evals_rhc, time_rhc)

# Simulated Annealing
sa_params = {'schedule': mlrose.ExpDecay(), 'max_attempts': 50, 'max_iters': 500, 'random_state': 42}
best_state_sa, best_fitness_sa, fitness_curve_sa, evals_sa, time_sa = run_algorithm(
    mlrose.simulated_annealing, problem, **sa_params)
results['Simulated Annealing'] = (best_state_sa, best_fitness_sa, fitness_curve_sa, evals_sa, time_sa)

# Genetic Algorithm
ga_params = {'pop_size': 100, 'mutation_prob': 0.2, 'max_iters': 500, 'random_state': 42}
best_state_ga, best_fitness_ga, fitness_curve_ga, evals_ga, time_ga = run_algorithm(
    mlrose.genetic_alg, problem, **ga_params)
results['Genetic Algorithm'] = (best_state_ga, best_fitness_ga, fitness_curve_ga, evals_ga, time_ga)

# Plot fitness by iteration for each algorithm separately
plt.figure(figsize=(12, 8))
plt.plot(results['Randomized Hill Climbing'][2], label='Randomized Hill Climbing')
plt.title('Fitness by Iteration for Randomized Hill Climbing')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(results['Simulated Annealing'][2], label='Simulated Annealing')
plt.title('Fitness by Iteration for Simulated Annealing')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(results['Genetic Algorithm'][2], label='Genetic Algorithm')
plt.title('Fitness by Iteration for Genetic Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()

# Report the number of function evaluations and wall clock time for each algorithm
for algo in results:
    print(f"{algo}:")
    print(f"  Best Fitness: {results[algo][1]}")
    print(f"  Number of Function Evaluations: {results[algo][3]}")
    print(f"  Wall Clock Time: {results[algo][4]:.4f} seconds\n")

# Save the modified DataFrame to a new CSV file
insurance_data['color_charges_rhc'] = best_state_rhc
insurance_data['color_charges_sa'] = best_state_sa
insurance_data['color_charges_ga'] = best_state_ga

output_file_path = 'modified_insurance_data_all_algorithms.csv'
insurance_data.to_csv(output_file_path, index=False)
print(f"Modified data saved to {output_file_path}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mlrose_hiive as mlrose
import time

# Load the data
data = pd.read_csv('insurance.csv')

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to print results
def print_results(algorithm_name, model, start_time, end_time, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    iterations = len(model.fitness_curve) if hasattr(model, 'fitness_curve') else "Not available"
    print(f"{algorithm_name} Mean Squared Error: {mse}")
    print(f"Wall Clock Time ({algorithm_name}): {end_time - start_time} seconds")
    print(f"Iterations ({algorithm_name}): {iterations}")
    print('-' * 40)

# Configurable parameter for the number of runs
num_rhc_runs = 5

# Define and train a neural network using backward propagation with mlrose
nn_model = mlrose.NeuralNetwork(hidden_nodes=[50],
                                activation='relu',
                                algorithm='gradient_descent',
                                max_iters=2000,
                                bias=True,
                                is_classifier=False,
                                learning_rate=1,
                                early_stopping=True,
                                clip_max=5,
                                max_attempts=100,
                                random_state=42,
                                curve=True)

start_time = time.time()
nn_model.fit(X_train, y_train)
end_time = time.time()
print_results("Backward Propagation with mlrose", nn_model, start_time, end_time, X_test, y_test)

# Randomized Hill Climbing - Multiple Runs
best_model = None
best_mse = float('inf')
best_iterations = 0
best_time = 0

for i in range(num_rhc_runs):
    nn_model = mlrose.NeuralNetwork(hidden_nodes=[50],
                                    activation='relu',
                                    algorithm='random_hill_climb',
                                    max_iters=2000,
                                    bias=True,
                                    is_classifier=False,
                                    learning_rate=1,
                                    early_stopping=True,
                                    clip_max=5,
                                    max_attempts=10,
                                    random_state=i,
                                    curve=True)

    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    y_pred = nn_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    iterations = len(nn_model.fitness_curve) if hasattr(nn_model, 'fitness_curve') else "Not available"

    if mse < best_mse:
        best_mse = mse
        best_model = nn_model
        best_iterations = iterations
        best_time = end_time - start_time

print(f"Best Randomized Hill Climbing Mean Squared Error: {best_mse}")
print(f"Best Wall Clock Time (Randomized Hill Climbing): {best_time} seconds")
print(f"Best Iterations (Randomized Hill Climbing): {best_iterations}")
print('-' * 40)

# Simulated Annealing
nn_model = mlrose.NeuralNetwork(hidden_nodes=[50],
                                activation='relu',
                                algorithm='simulated_annealing',
                                max_iters=2000,
                                bias=True,
                                is_classifier=False,
                                learning_rate=1,
                                early_stopping=True,
                                clip_max=5,
                                max_attempts=10,
                                random_state=42,
                                curve=True,
                                schedule=mlrose.ExpDecay(init_temp=5, exp_const=0.005))

start_time = time.time()
nn_model.fit(X_train, y_train)
end_time = time.time()
print_results("Simulated Annealing", nn_model, start_time, end_time, X_test, y_test)

# Genetic Algorithm
nn_model = mlrose.NeuralNetwork(hidden_nodes=[50],
                                activation='relu',
                                algorithm='genetic_alg',
                                max_iters=2000,
                                bias=True,
                                is_classifier=False,
                                learning_rate=1,
                                early_stopping=True,
                                clip_max=5,
                                max_attempts=10,
                                random_state=42,
                                curve=True,
                                mutation_prob=0.1)  # Adjust mutation probability

start_time = time.time()
nn_model.fit(X_train, y_train)
end_time = time.time()
print_results("Genetic Algorithm", nn_model, start_time, end_time, X_test, y_test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Global parameter to turn on/off plotting
PLOT_RESULTS = False

# Step 1: Load the dataset from the provided CSV file
file_path = 'insurance.csv'
data = pd.read_csv(file_path)

# Step 2: Prepare the data
# Assuming the target variable is 'charges'
target_column = "charges"
X = data.drop(target_column, axis=1)
y = data[target_column]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Step 3: Preprocess the data
# Create a column transformer to handle categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=[np.number]).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Function to plot learning curves
def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=None):
    if not PLOT_RESULTS:
        return
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), 
                                                            scoring='neg_mean_squared_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation error")
    
    plt.legend(loc="best")
    plt.show()

# Function to plot validation curves
def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=5, n_jobs=None):
    if not PLOT_RESULTS:
        return
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, 
                                                 cv=cv, n_jobs=n_jobs, scoring='neg_mean_squared_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Mean Squared Error")
    plt.grid()
    
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation error")
    
    plt.legend(loc="best")
    plt.show()

# Function to plot grid search results
def plot_grid_search(cv_results, grid_param, name_param):
    if not PLOT_RESULTS:
        return
    scores_mean = -cv_results['mean_test_score']
    
    plt.figure()
    plt.plot(grid_param, scores_mean, '-o')
    
    plt.title("Grid Search Scores")
    plt.xlabel(name_param)
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

# Function to train, evaluate, and plot learning and validation curves, including training time
def train_evaluate_plot(model, model_name, param_name, param_range, param_grid, plot_param_name=None):
    # Create a pipeline that first transforms the data and then fits the model
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and measure training time
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Make predictions on the training set
    y_train_pred = clf.predict(X_train)
    
    # Make predictions on the test set
    y_test_pred = clf.predict(X_test)
    
    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f'{model_name} Training Time: {training_time:.2f} seconds')
    print(f'{model_name} Training Mean Squared Error: {train_mse:.2f}')
    print(f'{model_name} Testing Mean Squared Error: {test_mse:.2f}')
    print(f'{model_name} Training R² Score: {train_r2:.2f}')
    print(f'{model_name} Testing R² Score: {test_r2:.2f}')
    print('---------------------------------------------')
    
    # Plot learning curve
    plot_learning_curve(clf, f"Learning Curves ({model_name})", X, y)
    
    # Plot validation curve for selected hyper-parameter
    if plot_param_name in param_name:
        idx = param_name.index(plot_param_name)
        plot_validation_curve(clf, f"Validation Curve ({model_name})", X, y, param_name[idx], param_range[idx])
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Plot grid search results
    plot_grid_search(grid_search.cv_results_, param_grid[param_name[0]], param_name[0])

# Step 6: Define and evaluate the models

# Neural Network Regressor with increased max_iter, learning_rate_init set to 1, and early stopping
nn_model = MLPRegressor(random_state=42, max_iter=2000, learning_rate_init=1, solver='adam', early_stopping=True, n_iter_no_change=10)
nn_param_name = ['regressor__learning_rate_init']
nn_param_range = [np.logspace(-5, -1, 5)]
nn_param_grid = {'regressor__learning_rate_init': np.logspace(-5, -1, 5)}

# k-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor()
knn_param_name = ['regressor__n_neighbors']
knn_param_range = [np.arange(1, 31)]
knn_param_grid = {'regressor__n_neighbors': np.arange(1, 31)}

# Support Vector Regressor
svr_model = SVR()
svr_param_name = ['regressor__C']
svr_param_range = [np.logspace(-3, 2, 6)]
svr_param_grid = {'regressor__C': np.logspace(-3, 2, 6)}

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(random_state=42, max_depth=1)
gbr_param_name = ['regressor__n_estimators']
gbr_param_range = [np.arange(50, 201, 50)]
gbr_param_grid = {'regressor__n_estimators': np.arange(50, 201, 50)}

# Example usage
# Set the parameter to plot (change this as needed)
plot_param_name = 'regressor__learning_rate_init'  # Change to the desired hyperparameter

# Train and evaluate the models
#train_evaluate_plot(nn_model, "Neural Network Regressor", nn_param_name, nn_param_range, nn_param_grid, plot_param_name)
#train_evaluate_plot(knn_model, "k-Nearest Neighbors Regressor", knn_param_name, knn_param_range, knn_param_grid, 'regressor__n_neighbors')
#train_evaluate_plot(svr_model, "Support Vector Regressor", svr_param_name, svr_param_range, svr_param_grid, 'regressor__C')
train_evaluate_plot(gbr_model, "Gradient Boosting Regressor", gbr_param_name, gbr_param_range, gbr_param_grid, 'regressor__n_estimators')

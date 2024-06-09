import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Global parameter to turn on/off plotting
PLOT_RESULTS = True

# Step 1: Load the dataset from the provided CSV file
file_path = 'smmh.csv'
data = pd.read_csv(file_path)

# Step 2: Prepare the data
# Assuming the target variable is '8. What is the average time you spend on social media every day?'
target_column = '8. What is the average time you spend on social media every day?'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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
                                                            scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation accuracy")
    
    plt.legend(loc="best")
    plt.show()

# Function to plot validation curves
def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=5, n_jobs=None):
    if not PLOT_RESULTS:
        return
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, 
                                                 cv=cv, n_jobs=n_jobs, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.grid()
    
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training accuracy")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation accuracy")
    
    plt.legend(loc="best")
    plt.show()

# Function to plot grid search results
def plot_grid_search(cv_results, grid_param, name_param):
    if not PLOT_RESULTS:
        return
    scores_mean = cv_results['mean_test_score']
    
    plt.figure()
    plt.plot(grid_param, scores_mean, '-o')
    
    plt.title("Grid Search Scores")
    plt.xlabel(name_param)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Function to train, evaluate, and plot learning and validation curves, including training time
def train_evaluate_plot(model, model_name, param_name, param_range, param_grid, plot_param_name=None):
    # Create a pipeline that first transforms the data and then fits the model
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
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
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f'{model_name} Training Time: {training_time:.2f} seconds')
    print(f'{model_name} Training Accuracy: {train_accuracy:.2f}')
    print(f'{model_name} Testing Accuracy: {test_accuracy:.2f}')
    print(f'{model_name} Classification Report:\n{classification_report(y_test, y_test_pred, zero_division=1)}')
    print('---------------------------------------------')
    
    # Plot learning curve
    plot_learning_curve(clf, f"Learning Curves ({model_name})", X, y)
    
    # Plot validation curve for selected hyper-parameter
    if plot_param_name in param_name:
        idx = param_name.index(plot_param_name)
        plot_validation_curve(clf, f"Validation Curve ({model_name})", X, y, param_name[idx], param_range[idx])
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Plot grid search results
    plot_grid_search(grid_search.cv_results_, param_grid[param_name[0]], param_name[0])

# Step 6: Define and evaluate the models

# Neural Network Classifier with increased max_iter, learning_rate_init set to 1, and early stopping
nn_model = MLPClassifier(random_state=42, max_iter=2000, learning_rate_init=1, solver='adam', early_stopping=True, n_iter_no_change=10)
nn_param_name = ['classifier__learning_rate_init']
nn_param_range = [np.logspace(-5, -1, 5)]
nn_param_grid = {'classifier__learning_rate_init': np.logspace(-5, -1, 5)}

# k-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier()
knn_param_name = ['classifier__n_neighbors']
knn_param_range = [np.arange(1, 31)]
knn_param_grid = {'classifier__n_neighbors': np.arange(1, 31)}

# Support Vector Classifier
svc_model = SVC()
svc_param_name = ['classifier__C']
svc_param_range = [np.logspace(-3, 2, 6)]
svc_param_grid = {'classifier__C': np.logspace(-3, 2, 6)}

# Gradient Boosting Classifier
gbc_model = GradientBoostingClassifier(random_state=42)
gbc_param_name = ['classifier__n_estimators']
gbc_param_range = [np.arange(50, 201, 50)]
gbc_param_grid = {'classifier__n_estimators': np.arange(50, 201, 50)}

# Example usage
# Set the parameter to plot (change this as needed)
plot_param_name = 'classifier__learning_rate_init'  # Change to the desired hyperparameter

# Train and evaluate the models
train_evaluate_plot(nn_model, "Neural Network Classifier", nn_param_name, nn_param_range, nn_param_grid, plot_param_name)
train_evaluate_plot(knn_model, "k-Nearest Neighbors Classifier", knn_param_name, knn_param_range, knn_param_grid, 'classifier__n_neighbors')
train_evaluate_plot(svc_model, "Support Vector Classifier", svc_param_name, svc_param_range, svc_param_grid, 'classifier__C')
train_evaluate_plot(gbc_model, "Gradient Boosting Classifier", gbc_param_name, gbc_param_range, gbc_param_grid, 'classifier__n_estimators')

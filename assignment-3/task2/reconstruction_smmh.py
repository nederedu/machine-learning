import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error
import sys
import os

def main(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display the first few rows to understand the structure
    print(data.head())

    # Identifying numeric and categorical columns
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Define the transformations for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder())
    ])

    # Combine the transformations using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply the transformations
    data_preprocessed = preprocessor.fit_transform(data)

    # Convert the sparse matrix to a dense array
    data_preprocessed = data_preprocessed.toarray()

    # Function to calculate reconstruction error for a given number of components
    def calculate_reconstruction_error(data, n_components):
        grp = GaussianRandomProjection(n_components=n_components, random_state=42)
        data_projected = grp.fit_transform(data)
        data_reconstructed = np.dot(data_projected, grp.components_)
        error = mean_squared_error(data, data_reconstructed)
        return error

    # Determine the optimal number of components
    errors = []
    components_range = range(1, data_preprocessed.shape[1] + 1)

    for n in components_range:
        error = calculate_reconstruction_error(data_preprocessed, n)
        errors.append(error)

    # Plot the reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.plot(components_range, errors, marker='o', linestyle='-', color='b')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    plt.xticks(components_range)
    
    # Save the plot with the dataset name in the file name
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    png_filename = f'reconstruction_error_{dataset_name}.png'
    plt.savefig(png_filename)
    plt.show()
    plt.close()

    print(f'Plot saved as: {png_filename}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        main(sys.argv[1])

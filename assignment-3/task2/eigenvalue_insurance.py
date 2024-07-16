import pandas as pd
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Extracting numeric columns only
    numeric_data = data.select_dtypes(include='number')

    # Handle missing values by filling with mean (or another strategy as needed)
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Standardize the dataset
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Apply PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Get eigenvalues (variance explained by each component)
    eigenvalues = pca.explained_variance_

    # Plot the distribution of eigenvalues
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    plt.title('Distribution of Eigenvalues')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.grid(True)

    # Extract the dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save the plot as a PNG file with the dataset name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, f'eigenvalues_plot_{dataset_name}.png')
    plt.savefig(plot_path)

    # Show the plot
    plt.show()

    print(f"Plot saved as: {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)

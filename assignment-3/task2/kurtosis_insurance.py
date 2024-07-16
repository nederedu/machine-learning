import sys
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from kneed import KneeLocator

def load_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Convert categorical columns to numeric using one-hot encoding
    categorical_cols = ['sex', 'smoker', 'region']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.fillna(df.mean())
    return df

def calculate_kurtosis_for_components(data, max_components):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kurtosis_values = []

    for n_components in range(1, max_components + 1):
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.01)
        try:
            components = ica.fit_transform(data_scaled)
            component_kurtosis = kurtosis(components, fisher=False).mean()
            kurtosis_values.append(component_kurtosis)
        except ConvergenceWarning:
            kurtosis_values.append(np.nan)  # Use NaN to indicate non-convergence

    return kurtosis_values

def plot_kurtosis(kurtosis_values, max_components, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), kurtosis_values, marker='o')
    plt.title('Average Kurtosis for Different Number of Components in ICA')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Kurtosis')
    plt.savefig(output_file)
    plt.show()

def find_optimal_components(kurtosis_values):
    x = range(1, len(kurtosis_values) + 1)
    y = kurtosis_values
    kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='increasing')
    return kneedle.elbow

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path)

    # Preprocess the data
    data_preprocessed = preprocess_data(data)

    # Determine the maximum number of components (you can adjust this based on your dataset)
    max_components = min(data_preprocessed.shape[1], 20)  # For example, set max to 20 or any reasonable number

    # Calculate kurtosis for different number of components
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        kurtosis_values = calculate_kurtosis_for_components(data_preprocessed, max_components)

    # Plot kurtosis values and save as PNG
    output_file = os.path.join(os.path.dirname(__file__), 'kurtosis_plot_insurance.png')
    plot_kurtosis(kurtosis_values, max_components, output_file)

    # Find the optimal number of components
    optimal_components = find_optimal_components(kurtosis_values)
    print(f"The optimal number of components is: {optimal_components}")

if __name__ == "__main__":
    main()

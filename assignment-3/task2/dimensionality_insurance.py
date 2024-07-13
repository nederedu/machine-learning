import sys
import os
import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    # Load the dataset
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Convert categorical columns to numeric using one-hot encoding
    categorical_cols = ['sex', 'smoker', 'region']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Fill remaining missing values
    df = df.fillna(df.mean())

    # Check if any NaN values remain
    if df.isnull().values.any():
        print("NaN values found in columns after preprocessing:")
        print(df.columns[df.isna().any()].tolist())
        print(df[df.isna().any(axis=1)])
        sys.exit(1)

    return df

def apply_randomized_projections(data, n_components):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    return rp.fit_transform(data)

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def apply_ica(data, n_components):
    ica = FastICA(n_components=n_components, random_state=42)
    return ica.fit_transform(data)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    data = load_dataset(file_path)

    # Preprocess the data
    data_preprocessed = preprocess_data(data)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_preprocessed)

    # Check for any remaining NaN values
    if np.any(np.isnan(data_scaled)):
        print("Data contains NaN values after preprocessing. Please check your data.")
        sys.exit(1)

    # Set the number of components to reduce to
    n_components = min(data_scaled.shape[1], 10)  # Adjust based on your needs

    # Apply dimensionality reduction techniques
    try:
        rp_result = apply_randomized_projections(data_scaled, n_components)
    except Exception as e:
        print(f"Error in Randomized Projections: {e}")
        sys.exit(1)

    try:
        pca_result = apply_pca(data_scaled, n_components)
    except Exception as e:
        print(f"Error in PCA: {e}")
        sys.exit(1)

    try:
        ica_result = apply_ica(data_scaled, n_components)
    except Exception as e:
        print(f"Error in ICA: {e}")
        sys.exit(1)

    # Create dataframes for the results
    rp_df = pd.DataFrame(rp_result, columns=[f'RP_{i+1}' for i in range(n_components)])
    pca_df = pd.DataFrame(pca_result, columns=[f'PCA_{i+1}' for i in range(n_components)])
    ica_df = pd.DataFrame(ica_result, columns=[f'ICA_{i+1}' for i in range(n_components)])

    # Save the results to CSV files
    rp_df.to_csv(f'{dataset_name}_randomized_projections.csv', index=False)
    pca_df.to_csv(f'{dataset_name}_pca_results.csv', index=False)
    ica_df.to_csv(f'{dataset_name}_ica_results.csv', index=False)

    print("Dimensionality reduction completed. Results saved to CSV files.")

if __name__ == "__main__":
    main()

import sys
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    le = LabelEncoder()
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    return df_processed

def perform_pca(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    return pca.fit_transform(scaled_data)

def apply_kmeans(data, n_clusters=3):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return kmeans.labels_, elapsed_time

def apply_em(data, n_components=3):
    start_time = time.time()
    em = GaussianMixture(n_components=n_components, random_state=42).fit(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return em.predict(data), elapsed_time

def plot_clusters(data, labels, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(filename)
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python clustering_script.py <insurance_csv_path> <smmh_csv_path>")
        sys.exit(1)

    insurance_csv_path = sys.argv[1]
    smmh_csv_path = sys.argv[2]

    insurance_df = load_data(insurance_csv_path)
    smmh_df = load_data(smmh_csv_path)

    insurance_df_processed = preprocess_data(insurance_df)

    if 'Timestamp' in smmh_df.columns:
        smmh_df = smmh_df.drop(columns=['Timestamp'])
    smmh_df_processed = preprocess_data(smmh_df)

    insurance_pca = perform_pca(insurance_df_processed)
    smmh_pca = perform_pca(smmh_df_processed)

    insurance_kmeans_labels, insurance_kmeans_time = apply_kmeans(insurance_pca)
    smmh_kmeans_labels, smmh_kmeans_time = apply_kmeans(smmh_pca)

    insurance_em_labels, insurance_em_time = apply_em(insurance_pca)
    smmh_em_labels, smmh_em_time = apply_em(smmh_pca)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    plot_clusters(insurance_pca, insurance_kmeans_labels, 'K-Means Clustering on Insurance Dataset',
                  os.path.join(script_dir, 'insurance_kmeans.png'))
    plot_clusters(smmh_pca, smmh_kmeans_labels, 'K-Means Clustering on SMMH Dataset',
                  os.path.join(script_dir, 'smmh_kmeans.png'))
    plot_clusters(insurance_pca, insurance_em_labels, 'EM Clustering on Insurance Dataset',
                  os.path.join(script_dir, 'insurance_em.png'))
    plot_clusters(smmh_pca, smmh_em_labels, 'EM Clustering on SMMH Dataset',
                  os.path.join(script_dir, 'smmh_em.png'))

    # Print wall clock times
    print(f"K-Means Clustering on Insurance Dataset took {insurance_kmeans_time:.4f} seconds")
    print(f"K-Means Clustering on SMMH Dataset took {smmh_kmeans_time:.4f} seconds")
    print(f"EM Clustering on Insurance Dataset took {insurance_em_time:.4f} seconds")
    print(f"EM Clustering on SMMH Dataset took {smmh_em_time:.4f} seconds")

if __name__ == "__main__":
    main()

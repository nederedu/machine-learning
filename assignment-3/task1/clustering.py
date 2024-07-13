import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA

# Check if the file path is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

# Get the file path from the command line argument
file_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(file_path))[0]

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Selecting numerical features for clustering
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
X = df[numerical_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Expectation Maximization (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, random_state=42) # Adjust the number of components if necessary
gmm_labels = gmm.fit_predict(X_scaled)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42) # Adjust the number of clusters if necessary
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataframe
df['GMM_Cluster'] = gmm_labels
df['KMeans_Cluster'] = kmeans_labels

# Display the first few rows of the dataframe with cluster labels
print(df.head())

# Plot the results of the clustering for the first two principal components (if applicable)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot for GMM Clustering
plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis')
plt.title('GMM Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
gmm_plot_path = f'{dataset_name}_GMM_Clustering.png'
plt.savefig(gmm_plot_path)
plt.show()

# Plot for K-Means Clustering
plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
kmeans_plot_path = f'{dataset_name}_KMeans_Clustering.png'
plt.savefig(kmeans_plot_path)
plt.show()

print(f"GMM Clustering plot saved as {gmm_plot_path}")
print(f"K-Means Clustering plot saved as {kmeans_plot_path}")

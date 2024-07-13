import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import sys

# Check if the file path is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

# Get the file path from the command line argument
file_path = sys.argv[1]

# Load the dataset
df = pd.read_csv(file_path)

# Selecting numerical features for clustering
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
X = df[numerical_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Expectation Maximization (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, random_state=42)  # Adjust the number of components if necessary
gmm_labels = gmm.fit_predict(X_scaled)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # Adjust the number of clusters if necessary
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataframe
df['GMM_Cluster'] = gmm_labels
df['KMeans_Cluster'] = kmeans_labels

# Save the new dataframe to a CSV file
output_file_path = 'smmh_with_clusters.csv'
df.to_csv(output_file_path, index=False)

print(f"The new dataset with clusters has been saved as {output_file_path}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Generate clustering dataset
X, y_true = make_blobs(
    n_samples=500, 
    centers=3,        
    cluster_std=1.0,  
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, edgecolors='k')
plt.title("Synthetic Clustering Dataset (3 blobs)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Kmeans model
k = 3
m, n = X.shape

def init_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    X_sq = np.sum(X**2, axis=1, keepdims=True)      
    C_sq = np.sum(centroids**2, axis=1, keepdims=True).T    
    cross = X @ centroids.T                                 
    distances_sq = X_sq + C_sq - 2 * cross  
    close_cents_value = np.min(distances_sq, axis=1)
    close_cents_index = np.argmin(distances_sq, axis=1)
    return close_cents_index, close_cents_value

def cost_function(close_cents_value):
    return np.mean(close_cents_value)

def update_centroids(X, labels, k):
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    counts = np.bincount(labels, minlength=k).reshape(-1, 1)
    np.add.at(centroids, labels, X)
    centroids = np.divide(centroids, counts, where=counts != 0)
    return centroids

centroids = init_centroids(X, k)

for _ in range(10):
    labels, close_cents_value = assign_clusters(X, centroids)
    centroids = update_centroids(X, labels, k)

plt.figure(figsize=(6, 6))
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=30, label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.legend()
plt.title("K-means Clustering Results")
plt.show()

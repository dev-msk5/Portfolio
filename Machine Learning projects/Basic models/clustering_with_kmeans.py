import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Generate clustering dataset
X, y_true = make_blobs(
    n_samples=500, 
    centers=3,        # number of clusters
    cluster_std=1.0,  # spread of clusters
    random_state=42
)

# 2. Plot dataset (ignore y_true, since clustering is unsupervised)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, edgecolors='k')
plt.title("Synthetic Clustering Dataset (3 blobs)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


#Kmeans model
X, y_true = make_blobs()
k = 3
m,n = X.shape
# print(X.shape, y_true.shape)
def init_centroids(X,k):
    centroids = X[np.random.choice(X.shape[0],k)]    # centroids from random points of given blobs
    return centroids

# print(init_centroids(X,3))
centroids = init_centroids(X,k)

# assign points to centroids
X_sq = np.sum(X**2, axis=1, keepdims=True)      
C_sq = np.sum(centroids**2, axis=1, keepdims=True).T    
cross = X @ centroids.T                                 
distances_sq = X_sq + C_sq - 2 * cross  
close_cents_value = np.min(distances_sq, axis=1)  

#TODO cost function
def cost_function(X,close_cents_value):
    return np.mean(close_cents_value)


#TODO move centroids to cluster means
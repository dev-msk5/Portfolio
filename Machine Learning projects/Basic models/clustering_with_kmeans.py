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
# print(X.shape, y_true.shape)
def init_centroids(X,k):
    centroids = X[np.random.choice(X.shape[0],k)]    # centroids from random points of given blobs
    return centroids

# print(init_centroids(X,3))
#TODO cost function

#TODO assign points to centroids

#TODO move centroids to cluster means
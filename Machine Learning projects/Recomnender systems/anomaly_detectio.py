import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# --- Normal data (950 points) ---
n_normal = 950
X_normal = np.random.randn(n_normal, 2)          # first two features
z_normal = np.random.randn(n_normal, 1) * 0.5   # third feature for normal cluster
X_normal = np.hstack([X_normal, z_normal])

# --- Anomalous data (50 points) ---
n_anomaly = 50
X_anomaly = np.random.randn(n_anomaly, 2)        # first two features overlap with normal
z_anomaly = np.random.uniform(5, 7, size=(n_anomaly, 1))  # third feature far away
X_anomaly = np.hstack([X_anomaly, z_anomaly])

# --- Combine ---
X = np.vstack([X_normal, X_anomaly])
y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])  # 0=normal, 1=anomaly

# Shuffle dataset
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

# --- 3D Plot ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y==0][:,0], X[y==0][:,1], X[y==0][:,2], c='blue', label='Normal', alpha=0.6)
ax.scatter(X[y==1][:,0], X[y==1][:,1], X[y==1][:,2], c='red', label='Anomaly', alpha=0.8)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("3D Synthetic Anomaly Detection Dataset")
ax.legend()
plt.show()


# Anomaly detection model
sigma = np.std(X, axis=0)
mu = np.mean(X, axis=0)
epsilon =  10**-X.shape[1]


def product(X,mean,deviation):
    px = (1/np.sqrt(2*np.pi*deviation))*np.exp((-(X-mean)**2)/(2*deviation**2))
    probabilities = np.prod(px,axis=1)
    return probabilities

def anomaly_or_not(probabilities,threshold):
    return threshold > probabilities

mask = anomaly_or_not(product(X,mu,sigma), epsilon)
anomalies = X[mask]
normal = X[~mask]

# plot
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal[:,0], normal[:,1], normal[:,2],
           c='blue', label='Normal', alpha=0.6)
ax.scatter(anomalies[:,0], anomalies[:,1], anomalies[:,2],
           c='red', label='Anomaly', alpha=0.9)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("3D Anomaly Detection Results")
ax.legend()
plt.show()
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def generate_cf_data(n_users=20, n_items=15, latent_dim=3, sparsity=0.3, seed=42):
    """
    Generate synthetic user–item rating matrix for collaborative filtering.

    Args:
        n_users (int): number of users
        n_items (int): number of items
        latent_dim (int): number of latent factors
        sparsity (float): fraction of missing values (0=full, 1=empty)
        seed (int): reproducibility

    Returns:
        R (ndarray): rating matrix with missing entries = 0
        mask (ndarray): 1 if rating is observed, 0 if missing
        R_full (ndarray): full "true" ratings before masking
        U (ndarray): user weight vectors (n_users × latent_dim)
        V (ndarray): item weight vectors (n_items × latent_dim)
    """
    np.random.seed(seed)

    # User & item latent vectors
    U = np.random.normal(0, 1, (n_users, latent_dim))   # user weights/preferences
    V = np.random.normal(0, 1, (n_items, latent_dim))   # item latent features

    # Ratings as dot product of user and item features
    R_full = U @ V.T

    # Normalize ratings to 1–5
    R_full = 1 + 4 * (R_full - R_full.min()) / (R_full.max() - R_full.min())

    # Mask (sparsity)
    mask = np.random.rand(n_users, n_items) > sparsity
    R = R_full * mask

    return R, mask, R_full, U, V


# Example usage
X, mask, R_full, W, V = generate_cf_data(n_users=5, n_items=6, latent_dim=3, sparsity=0.4)


# model
num_movies, num_features = X.shape
num_users = R_full.shape[1]
X = tf.Variable(tf.random.normal((num_movies, num_features)), dtype=tf.float32)
W = tf.Variable(tf.random.normal((num_users, num_features)), dtype=tf.float32)
b = tf.Variable(tf.zeros((1,)), dtype=tf.float32)

# Dummy observed ratings and mask
Y = tf.random.uniform((num_movies, num_users), minval=1, maxval=5, dtype=tf.float32)
R = tf.cast(tf.random.uniform((num_movies, num_users), minval=0, maxval=2, dtype=tf.int32), tf.float32)

lambda_ = 0.07
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 500
costs=[]

for i in range(iterations):
    with tf.GradientTape() as tape:
        preds = tf.matmul(X, W, transpose_b=True) + b   # shape (movies, users)
        error = (preds - Y) * R                         # mask missing ratings
        cost = 0.5 * tf.reduce_sum(tf.square(error))    # squared error
        cost += (lambda_/2) * (tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(X)))  # regularization

    grads = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if i % 20 == 0:
        print(f"Iteration {i}, cost = {cost.numpy():.4f}")
        costs.append(cost)


plt.plot(range(int(iterations/20)), costs, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Collaborative Filtering Training Cost')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def generate_circles(n_samples=1000, noise=0.1):
    angles = 2 * np.pi * np.random.rand(n_samples)
    radii = np.sqrt(np.random.rand(n_samples))

    X_inner = np.c_[radii*np.cos(angles), radii*np.sin(angles)]
    y_inner = np.zeros(n_samples)

    X_outer = np.c_[(radii+1.5)*np.cos(angles), (radii+1.5)*np.sin(angles)]
    y_outer = np.ones(n_samples)

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])

    # add some noise
    X += noise * np.random.randn(*X.shape)

    return X, y

# Example
X, y = generate_circles()
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
plt.title("Circles dataset (good for sigmoid NN)")
plt.show()


# Neural network model
print(X.shape, y.shape)
m, n = X.shape

# 1st layer, 4 neurons
W_1 = np.random.randn(n, 4) * 0.01  # *0.01 avoids symmetry problems
b_1 = np.zeros((1, 4))

# 2nd layer, 1 neuron
W_2 = np.random.randn(4, 1) * 0.01  # *0.01 avoids symmetry problems
b_2 = np.zeros((1, 1))

def sigmoid(Z):
    g = 1 / (1 + np.exp(-Z))
    return g

def Dense(A_in,W,B):
    Z = np.matmul(A_in,W) + B.reshape(1, -1)
    A_out = sigmoid(Z)
    return A_out

def Sequential(X):
    a1 = Dense(X,W_1,b_1)
    a2 = Dense(a1,W_2,b_2)
    return a2

# Plot the neural network output
y_pred = Sequential(X)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_pred.flatten(), cmap="coolwarm", alpha=0.7)
plt.title("Neural Network Output")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="NN Output")
plt.show()

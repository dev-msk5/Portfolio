import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

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


# # Load digits dataset (8x8 images, labels 0–9)
# digits = load_digits()

# # Select only 500 samples for training
# X, y = digits.data[:500], digits.target[:500]   # X shape (500, 64)

# # Scale input features (important for NN training)
# X = X / 16.0  # since pixel values range 0–16

# # Convert multi-class y (0–9) -> binary labels (0 if <5, 1 if >=5)
# y = (y >= 5).astype(int)  # shape (500,)

# print("X shape:", X.shape)   # (500, 64)
# print("y shape:", y.shape)   # (500,)


# Example
X, y = generate_circles()
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
plt.title("Circles dataset (good for sigmoid NN)")        ## 1st dataset
plt.show()


# Neural network model
# print(X.shape, y.shape)
m, n = X.shape
lambda_ = 0.1
epochs = 250
alpha = 0.001

# 1st layer, 4 neurons
W_1 = np.random.randn(n, 16) * 0.01     # 0.01 avoids symmetry problems
b_1 = np.zeros((1, 16))


# 2nd layer, 1 neuron
W_2 = np.random.randn(16, 1) * 0.01
b_2 = np.zeros((1, 1))

def sigmoid(Z):
    g = 1 / (1 + np.exp(-Z))
    return g

def cost_function(a2,W_1,W_2,y,lambda_):
    z = a2
    regularization_term = lambda_ * (np.sum(W_1 ** 2) + np.sum(W_2 ** 2)) / (2 * m)         # 2 layer -> 2 W
    cost = ((np.sum(y * np.log(z + 1e-8) + (1 - y) * np.log(1 - z + 1e-8)) / -m) + regularization_term)
    return cost

def gradient_function_2(a1, a2, y, lambda_, W_2):
    error = a2 - y.reshape(-1, 1)
    dJ_dW_2 = (np.matmul(a1.T,error)) / m + lambda_ * W_2 / m
    dJ_db_2 = np.sum(error, axis=0, keepdims=True) / m
    return dJ_dW_2, dJ_db_2

def gradient_function_1(X, a1, a2, y, lambda_, W_1, b_1, W_2):
    error = a2 - y.reshape(-1, 1)
    dZ1 = (np.matmul(error,W_2.T)) * a1 * (1 - a1)  # sigmoid derivative
    dJ_dW_1 = (np.matmul(X.T,dZ1)) / m + (lambda_ / m) * W_1
    dJ_db_1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dJ_dW_1, dJ_db_1

def Dense(A_in,W,B):
    Z = np.matmul(A_in,W) + B.reshape(1, -1)
    A_out = sigmoid(Z)
    return A_out

def Sequential(X):
    a1 = Dense(X,W_1,b_1)
    a2 = Dense(a1,W_2,b_2)

    return a2

# Plot the neural network output
losses = []
def neural_net(X, W_1, y, lambda_, b_1, W_2):
    global b_2
    for i in range(epochs):
        a1 = Dense(X, W_1, b_1)
        a2 = Dense(a1, W_2, b_2)

        loss = cost_function(a2, W_1, W_2, y, lambda_)
        losses.append(loss)

        dJ_dW_1, dJ_db_1 = gradient_function_1(X, a1, a2, y, lambda_, W_1, b_1, W_2)
        dJ_dW_2, dJ_db_2 = gradient_function_2(a1, a2, y, lambda_, W_2)

        W_1 -= alpha * dJ_dW_1
        b_1 -= alpha * dJ_db_1
        W_2 -= alpha * dJ_dW_2
        b_2 -= alpha * dJ_db_2

    return a2

y_pred = neural_net(X, W_1, y, lambda_, b_1, W_2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred.flatten(), cmap="coolwarm")                    ## 1st dataset
plt.title("Neural Network Output")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="NN Output")
plt.show()

# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.show()


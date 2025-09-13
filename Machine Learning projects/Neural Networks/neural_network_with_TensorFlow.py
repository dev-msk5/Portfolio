import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

def generate_xor(n_samples=1000, noise=0.1):
    X = np.random.randn(n_samples, 2)
    y = np.logical_xor(X[:,0] > 0, X[:,1] > 0).astype(int)
    
    X += noise * np.random.randn(*X.shape)
    return X, y

# Example
X, y = generate_xor()
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
plt.title("XOR dataset")    
plt.show()

model = Sequential([
    Dense(units=5, activation="relu"),
    Dense(units=1, activation="sigmoid")
])
model.compile(loss=BinaryCrossentropy())
model.fit(X,y,epochs=100)

def plot_decision_boundary(model, X, y, steps=200, cmap="coolwarm"):
    # Create a grid over feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                         np.linspace(y_min, y_max, steps))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid, verbose=0)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k")
    plt.title("Decision Boundary (XOR)")
    plt.show()

plot_decision_boundary(model, X, y)

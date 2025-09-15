import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

def generate_spiral(n_points=1000, n_classes=3, noise=0.2):
    X = []
    y = []
    for j in range(n_classes):
        ix = range(n_points*j, n_points*(j+1))
        r = np.linspace(0.0, 1, n_points)  # radius
        t = np.linspace(j*4, (j+1)*4, n_points) + np.random.randn(n_points)*noise
        X.extend(np.c_[r*np.sin(t), r*np.cos(t)])
        y.extend([j]*n_points)
    return np.array(X), np.array(y)

# Example
X, y = generate_spiral(n_points=300, n_classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")
plt.title("Spiral dataset (good for softmax NN)")
plt.show()

# Softmax model
model = Sequential([
    Dense(units=8, activation="relu"),
    Dense(units=4, activation="relu"),
    Dense(units=3, activation="linear")     # softmax correct syntax is linear with logits = True
])

model.compile(optimizer=Adam(learning_rate=3e-3),loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X,y,epochs=150)

def plot_decision_boundary(model, X, y):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict class probabilities for each point in the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid, verbose=0)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)  # take class with max probability

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Model decision boundary")
    plt.show()
plot_decision_boundary(model, X, y)
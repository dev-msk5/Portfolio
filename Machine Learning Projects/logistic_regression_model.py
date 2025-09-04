import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# Logistic regression dataset (binary classification, 2D for visualization)
def generate_logistic_data(n_samples=100):
    X = np.random.randn(n_samples, 2)
    # true weights
    w = np.array([[2.0], [-3.0]])
    b = 0.5
    logits = X @ w + b
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    return X, y, w, b
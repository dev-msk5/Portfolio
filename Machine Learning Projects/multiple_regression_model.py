import numpy as np
import matplotlib.pyplot as plt
import math
import copy


# Multiple linear regression dataset (y = Xw + b + noise)
def generate_multiple_linear_data(n_samples=100, n_features=3, noise=1.0):
    X = np.random.rand(n_samples, n_features) * 10
    true_w = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    y = X @ true_w + b + np.random.randn(n_samples, 1) * noise
    return X, y, true_w, b
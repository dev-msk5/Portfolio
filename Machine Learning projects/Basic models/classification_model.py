import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# Classification dataset (multi-class, 3 classes in 2D)
def generate_classification_data(n_samples=300, n_classes=3):
    X = []
    y = []
    for class_idx in range(n_classes):
        # Create points around a circle (different mean per class)
        theta = np.linspace(0, 2*np.pi/n_classes, n_samples//n_classes) + (class_idx*2*np.pi/n_classes)
        r = 5 + np.random.randn(n_samples//n_classes)
        x1 = r * np.cos(theta) + np.random.randn(n_samples//n_classes) * 0.5
        x2 = r * np.sin(theta) + np.random.randn(n_samples//n_classes) * 0.5
        X.append(np.c_[x1, x2])
        y.append(np.ones(n_samples//n_classes, dtype=int) * class_idx)
    return np.vstack(X), np.hstack(y)
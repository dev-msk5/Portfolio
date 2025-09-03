# The aim of this project is to start my ML journey. After I learned the concepts of these model, it's time to actually build them.
# Only using numpy, libraries like Tensorflow and Pandas will be used later down the road.

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# Linear regression dataset (y = ax + b + noise)
def generate_linear_data(n_samples=100, a=2.0, b=5.0, noise=1.0):
    X = np.random.rand(n_samples, 1) * 10  # X between 0 and 10
    y = a * X + b + np.random.randn(n_samples, 1) * noise
    return X, y

# Multiple linear regression dataset (y = Xw + b + noise)
def generate_multiple_linear_data(n_samples=100, n_features=3, noise=1.0):
    X = np.random.rand(n_samples, n_features) * 10
    true_w = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    y = X @ true_w + b + np.random.randn(n_samples, 1) * noise
    return X, y, true_w, b

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



#Linear regression
X, y = generate_linear_data()
w=100
b=100
alfa=0.03
iterations=5000

def compute_cost(X,y,w,b):      # Computing cost of a dataset
    
    m = X.shape[0]
    fwb = X * w + b
    cost = (np.sum((fwb - y) ** 2)/ (2 * m))  
    
    return cost

def gradient_function(X,y,w,b):         # Computing gradient for gradient descent, 1 iteration
    m=X.shape[0]
    dj_db=0
    dj_dw=0

    for i in range(m):
        fwb_i= np.dot(w,X[i])+b
        dj_dw += (fwb_i-y[i])*X[i]
        dj_db += fwb_i-y[i]
    dj_dw/=m
    dj_db/=m
    
    return dj_dw,dj_db


def linear_model(X,y,w,b,alfa,iterations):
    print(X[:5])
    print(y[:5])
    w = copy.deepcopy(w)
    w_history=[]
    J_history=[]
    
    
    for i in range(iterations):                 # - Implementing gradient descent -
        
        dj_dw, dj_db =gradient_function(X,y,w,b)                
        b -= alfa * dj_db
        w -= alfa * dj_dw                                            # - 
    
        if i<10000:                         # Stopping if too many iterations
            cost=compute_cost(X,y,w,b)
        else:
            break
        
        if i% math.ceil(iterations/10) == 0:            # Saving data / 10% steps
            w_history.append(w)
            J_history.append(cost)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f} ")
    
    m = X.shape[0]        
    prediction = np.zeros(m)

    for i in range(m):
        prediction[i] = (w * X[i] + b).item()
    
    print(f"Final weight: {float(w):8.4f},     final bias: {float(b):8.4f}")        
    # Plot the linear fit
    plt.plot(X, prediction, c = "b")

    # Create a scatter plot of the data
    plt.scatter(X, y, marker='x', c='r') 
    
    plt.title("Prediction of model")
    plt.ylabel('Generated output')      # Plot axes
    plt.xlabel('Generated input')
    plt.show()
    

    return w, b, J_history, w_history

linear_model(X,y,w,b,alfa,iterations)
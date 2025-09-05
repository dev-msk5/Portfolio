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

#Logistic regression model

X, y, true_W, true_b = generate_logistic_data()
W = np.ones((X.shape[1],1))
b = 1
m,n = X.shape

def sigmoid(X,W,b):
    z = np.dot(X,W) + b
    fwb = (1 / (1 + np.exp(-z)))
    return fwb
# print(sigmoid(X,W,b).shape)

def cost_function(X,y,W,b):
    fwb = sigmoid(X,W,b)        # (100, 1) , same shape as y
    return np.sum(y*np.log(fwb)+(1-y)*np.log(1-fwb)) / - m     # cost function using simplified definition

# print(cost_function(X,y,W,b))

def gradient_function(X,y,W,b):
    
    dj_dW = np.zeros(n)
    dj_db = 0
    
    error = sigmoid(X,W,b) - y
    dj_dW = (np.dot(X.T,error) / m)
    dj_db = (np.sum(error) / m)
    
    return dj_dW, dj_db
    

def logistic_regression_model(X,y,W,b):
    pass
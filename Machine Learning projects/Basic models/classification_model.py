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


# Classification with logistic regression model
X, y = generate_classification_data()
# print(X[:],y[:])
X1X2 = X[:,0] * X[:,1]
X = np.column_stack((X, X1X2))
m, n = X.shape
W = np.ones((n,1))
b = 1
alfa=0.01
iterations = 5000
lambda_ = 0.1

def sigmoid(X,W,b):
    z = np.dot(X,W) + b
    fwb = (1 / (1 + np.exp(-z)))
    return fwb

def compute_cost(X,y,W,b,lambda_):
    fwb = sigmoid(X,W,b)    # (300, 1)
    fwb = np.clip(fwb, 1e-8, 1 - 1e-8)  # Prevent log(0)
    y = y.reshape(-1, 1)    # Ensure y is a column vector to match fwb shape
    regularization_term = lambda_*np.dot(W.T,W)/ (2*m)
    cost = ((np.sum(y*np.log(fwb)+(1-y)*np.log(1-fwb)) / - m) + regularization_term)    # cost function using simplified definition
    return cost  

def gradient_function(X,y,W,b,lambda_):
    y = y.reshape(-1, 1)    # Ensure y is a column vector to match predictions
    error = sigmoid(X,W,b) - y 
    dj_dW = (np.dot(X.T,error) / m) + lambda_* W / m
    
    dj_db = np.sum(error) / m
    return dj_dW, dj_db
    
def classification_with_log_regr(X,y,W,b,alfa,iterations,lambda_):
    W = copy.deepcopy(W)
    W_history=[]
    J_history=[]
    
    for i in range(iterations):
        dj_dW, dj_db = gradient_function(X,y,W,b,lambda_)
        W -= alfa * dj_dW
        b -= alfa * dj_db
        
        if i<10000:                         # Stopping if too many iterations
            cost=compute_cost(X,y,W,b,lambda_)
        else:
            break
        
        if i% math.ceil(iterations/20) == 0:            # Saving data / 5% steps
            W_history.append(W)
            J_history.append(float(cost))
            print(f"Iteration {i:4}: Cost {J_history[-1]:8.2f} ")
            
    # 2D plot
    plt.figure(figsize=(8,6))
    colors = ['red', 'green', 'blue']
    for class_idx in range(3):
        plt.scatter(X[y==class_idx, 0], X[y==class_idx, 1], c=colors[class_idx], label=f'Class {class_idx}', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Classification Results')
    plt.legend()
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for class_idx in range(3):
        ax.scatter(X[y==class_idx, 0], X[y==class_idx, 1], X[y==class_idx, 2], 
                   c=colors[class_idx], label=f'Class {class_idx}', alpha=0.6)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3 (X1*X2)')
    ax.set_title('3D Classification Results')
    ax.legend()
    plt.show()
            
    return W, b, J_history, W_history

classification_with_log_regr(X,y,W,b,alfa,iterations,lambda_)
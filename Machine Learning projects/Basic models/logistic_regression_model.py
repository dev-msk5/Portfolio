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

# Logistic regression model

X, y, true_W, true_b = generate_logistic_data()
W = np.ones((X.shape[1],1))
b = 1
alfa=0.01
iterations=5000
lambda_=0.1
m,n = X.shape

def sigmoid(X,W,b):
    z = np.dot(X,W) + b
    fwb = (1 / (1 + np.exp(-z)))
    return fwb
# print(sigmoid(X,W,b).shape)

def compute_cost(X,y,W,b,lambda_):
    fwb = sigmoid(X,W,b)        # (100, 1) , same shape as y
    regularization_term = lambda_*np.dot(W.T,W) / (2*m)
    return ((np.sum(y*np.log(fwb)+(1-y)*np.log(1-fwb)) / - m) + regularization_term)    # cost function using simplified definition

# print(cost_function(X,y,W,b))

def gradient_function(X,y,W,b,lambda_):
    
    dj_dW = np.zeros(n)
    dj_db = 0
    
    error = sigmoid(X,W,b) - y
    dj_dW = (np.dot(X.T,error) / m) + (lambda_*W / m)
    dj_db = (np.sum(error) / m)
    
    return dj_dW, dj_db
    

def logistic_regression_model(X,y,W,b,alfa,iterations):
    print(X[:5])
    print(y[:5])
    W = copy.deepcopy(W)
    W_history=[]
    J_history=[]
    
    for i in range(iterations):
        dj_dW, dj_db =gradient_function(X,y,W,b,lambda_)                
        W -= alfa*dj_dW
        b -= alfa*dj_db
    
        if i<10000:                         # Stopping if too many iterations
            cost=compute_cost(X,y,W,b,lambda_)
        else:
            break
        
        if i% math.ceil(iterations/20) == 0:            # Saving data / 5% steps
            W_history.append(W)
            J_history.append(float(cost))
            print(f"Iteration {i:4}: Cost {J_history[-1]:8.2f} ")
            
    plt.figure(figsize=(8,6))
    # Plot data points
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='bwr', edgecolor='k', alpha=0.7)
    # Decision boundary: W1*x1 + W2*x2 + b = 0 => x2 = -(W1*x1 + b)/W2
    x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    x2_vals = -(W[0]*x1_vals + b)/W[1]
    plt.plot(x1_vals, x2_vals.flatten(), 'g-', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

    # Plot useful ML metrics: Cost vs Iterations
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(J_history)), J_history, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.title('Cost vs Iterations')
    plt.grid(True)
    plt.show()
            
    return W, b, J_history, W_history

logistic_regression_model(X,y,W,b,alfa,iterations)
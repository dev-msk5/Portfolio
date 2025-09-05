import numpy as np
import matplotlib.pyplot as plt
import math
import copy


# Multiple linear regression dataset (y = XW + b + noise)
def generate_multiple_linear_data(n_samples=100, n_features=4, noise=1.05):
    X = np.random.rand(n_samples, n_features) * 10
    true_W = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    y = X @ true_W + b + np.random.randn(n_samples, 1) * noise
    return X, y, true_W, b


# Multiple Variable Linear Regression
X, y, true_W, true_b =generate_multiple_linear_data()
print(f"X Shape: {X.shape},  \n{X[:5]}")
print(f"y shape: {y.shape},  \n{y[:5]}")

b=100
n_features = X.shape[1]   # number of features
W = np.full((n_features, 1), 10.0)     # making weight vector = 10 for every element
alfa = 0.015
iterations = 8000


def compute_cost(X,y,W,b):  # cost function 
    m = X.shape[0]
    
    fwb= np.dot(X,W)+b
    cost = (np.sum((fwb - y) ** 2)/ (2 * m))
    
    return cost

def gradient_function(X,y,W,b): # gradient function
    m,n = X.shape   # m roWs or examples , n colums or features
    
    dj_dW=np.zeros((n,))
    dj_db=0
    
    fwb = np.dot(X,W)+b
    error = (fwb-y)
    dj_dW = (np.dot(X.T,error)/m)   # n x 1 column vector
    dj_db = (np.sum(error)/m)       # 1 x 1 scalar
    
    return dj_dW, dj_db

def multiple_linear_model(X,y,W,b,alfa,iterations):
    print(X[:5])
    print(y[:5])
    W = copy.deepcopy(W)
    W_history=[]
    J_history=[]
    
    
    for i in range(iterations):                 # - Implementing gradient descent -
        
        dj_dW, dj_db =gradient_function(X,y,W,b)                
        b -= alfa * dj_db
        W -= alfa * dj_dW                                            # - 
    
        if i<10000:                         # Stopping if too many iterations
            cost=compute_cost(X,y,W,b)
        else:
            break
        
        if i% math.ceil(iterations/20) == 0:            # Saving data / 5% steps
            W_history.append(W)
            J_history.append(cost)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f} ")
    
    
    # print(f"{'Final bias':>15} | {'True bias':>15}")
    # print(f"{float(b):15.4f} | {float(true_b):15.4f}\n")
    print(f"{'Final weights':>15} | {'True weights':>15}")
    for fw, tw in zip(W.flatten(), true_W.flatten()):
        print(f"{fw:15.4f} | {tw:15.4f}")
    # Plot the linear fit
    plt.plot(range(len(J_history)), J_history, c="b")
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    

    return W, b, J_history, W_history

multiple_linear_model(X,y,W,b,alfa,iterations)
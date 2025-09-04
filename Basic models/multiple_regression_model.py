import numpy as np
import matplotlib.pyplot as plt
import math
import copy


# Multiple linear regression dataset (y = Xw + b + noise)
def generate_multiple_linear_data(n_samples=100, n_features=4, noise=1.05):
    X = np.random.rand(n_samples, n_features) * 10
    true_w = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    y = X @ true_w + b + np.random.randn(n_samples, 1) * noise
    return X, y, true_w, b


# Multiple Variable Linear Regression
X, y, true_w, true_b =generate_multiple_linear_data()
print(f"X Shape: {X.shape},  \n{X[:5]}")
print(f"y shape: {y.shape},  \n{y[:5]}")

b=100
n_features = X.shape[1]   # number of features
w = np.full((n_features, 1), 10.0)     # making weight vector = 10 for every element
alfa = 0.015
iterations = 8000


def compute_cost(X,y,w,b):  # cost function 
    m = X.shape[0]
    
    fwb= np.dot(X,w)+b
    cost = (np.sum((fwb - y) ** 2)/ (2 * m))
    
    return cost

def gradient_function(X,y,w,b): # gradient function
    m,n = X.shape   # m rows or examples , n colums or features
    
    dj_dw=np.zeros((n,))
    dj_db=0
    
    fwb = np.dot(X,w)+b
    error = (fwb-y)
    dj_dw = (np.dot(X.T,error)/m)   # n x 1 column vector
    dj_db = (np.sum(error)/m)       # 1 x 1 scalar
    
    return dj_dw, dj_db

def multiple_linear_model(X,y,w,b,alfa,iterations):
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
    
    
    # print(f"{'Final bias':>15} | {'True bias':>15}")
    # print(f"{float(b):15.4f} | {float(true_b):15.4f}\n")
    print(f"{'Final weights':>15} | {'True weights':>15}")
    for fw, tw in zip(w.flatten(), true_w.flatten()):
        print(f"{fw:15.4f} | {tw:15.4f}")
    # Plot the linear fit
    plt.plot(range(len(J_history)), J_history, c="b")
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    

    return w, b, J_history, w_history

multiple_linear_model(X,y,w,b,alfa,iterations)
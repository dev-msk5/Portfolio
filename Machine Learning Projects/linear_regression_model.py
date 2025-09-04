import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# Linear regression dataset (y = ax + b + noise)
def generate_linear_data(n_samples=100, a=2.0, b=5.0, noise=1.0):
    X = np.random.rand(n_samples, 1) * 10  # X between 0 and 10
    y = a * X + b + np.random.randn(n_samples, 1) * noise
    return X, y


#Linear regression model
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
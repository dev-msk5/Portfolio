import torch
from torch import nn

weight_setup = 0.7
bias_setup = 2
data = torch.randn(50)
y_setup = weight_setup * data + bias_setup

# print(data)
X_train, y_train = data[:len(data)*0.8], y_setup[:len(data)*0.8]
X_test,y_test = data[len(data)*0.8:], y_setup[len(data)*0.8:]

# Linear regression model
class Linear_Regression_Model():
    def __init__(self):
        super.__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
        
    def forward(self, X):
        return self.weights * X + self.bias
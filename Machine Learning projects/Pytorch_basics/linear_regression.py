import torch
from torch import nn
import matplotlib.pyplot as plt

weight_setup = 1.618
bias_setup = 42
data = torch.randn(100)
y_setup = weight_setup * data + bias_setup

# print(data)
X_train, Y_train = data[:int(len(data)*0.8)], y_setup[:int(len(data)*0.8)]
X_test, Y_test = data[int(len(data)*0.8):], y_setup[int(len(data)*0.8):]

# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, X):
        return self.weights * X + self.bias

model = LinearRegressionModel()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.04)
loss_fn = nn.MSELoss()

epochs = 5000

# Training
for epoch in range(epochs):
    model.train() #Training mode
    pred = model(X_train) #Training on data

    loss = loss_fn(pred, Y_train)  # Loss function

    optimizer.zero_grad() # Zeroing grad for current iteration
    loss.backward()
    optimizer.step()

model.eval()

with torch.inference_mode():
    prediction_test = model(X_test)

# Plots
print(f"Final weight: {model.weights.item():8.4f},     final bias: {model.bias.item():8.4f}")        
# Plot the linear fit
plt.plot(X_test, prediction_test, c = "b")

# Create a scatter plot of the data
plt.scatter(X_test, Y_test, marker='x', c='r') 

plt.title("Prediction of model")
plt.ylabel('Generated output')      # Plot axes
plt.xlabel('Generated input')
plt.show()




import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path


BASE = Path(__file__).resolve().parent

train = pd.read_csv(BASE/"train.csv")
test = pd.read_csv(BASE/"test.csv")

#Cleaning
features = ["HomePlanet","CryoSleep","Destination","Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
train_x_df = train[features].copy()
train_y_df = train["Transported"].copy()
test_x_df = test[features].copy()

#Making categorical numeric data for features
train_x_df["VIP"] = train_x_df["VIP"].map({False:0, True:1, 'False':0, 'True':1})
# print(train_x_df["HomePlanet"].unique())
train_x_df["CryoSleep"] = train_x_df["CryoSleep"].map({False:0, True:1, 'False':0, 'True':1})
train_x_df["HomePlanet"] = train_x_df["HomePlanet"].map({'Europa':0, 'Earth':1,'Mars':2})
train_x_df["Destination"] = train_x_df["Destination"].map({'TRAPPIST-1e':0, 'PSO J318.5-22':1,'55 Cancri e':2})

test_x_df["VIP"] = test_x_df["VIP"].map({False:0, True:1, 'False':0, 'True':1})
# print(test_x_df["HomePlanet"].unique())
test_x_df["CryoSleep"] = test_x_df["CryoSleep"].map({False:0, True:1, 'False':0, 'True':1})
test_x_df["HomePlanet"] = test_x_df["HomePlanet"].map({'Europa':0, 'Earth':1,'Mars':2})
test_x_df["Destination"] = test_x_df["Destination"].map({'TRAPPIST-1e':0, 'PSO J318.5-22':1,'55 Cancri e':2})

#For targets 
train_y_df = train_y_df.map({False:0, True:1, 'False':0, 'True':1})
print(f"Shape of training samples before removing NaN: {train_x_df.shape}")
print(f"Shape of test samples before removing NaN: {test_x_df.shape}")


# drop rows with any NaN in features and align targets
valid_idx = train_x_df.dropna().index
train_x_df = train_x_df.loc[valid_idx].reset_index(drop=True)
train_y_df = train_y_df.loc[valid_idx].reset_index(drop=True)
print(f"Shape of training samples after removing NaN: {train_x_df.shape}")
print(f"Shape of test samples after removing NaN: {train_y_df.shape}")


#Normalize data
X_tensor = torch.tensor(train_x_df.values.astype(np.float32))
y_tensor = torch.tensor(train_y_df.values.astype(np.float32)).unsqueeze(1)
train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

# fill missing values in test set using training feature means to avoid NaNs
test_filled = test_x_df.fillna(train_x_df.mean())
test_X_tensor = torch.tensor(test_filled.values.astype(np.float32))

# compute mean/std and normalize tensors
mean = X_tensor.mean(dim=0, keepdim=True)
std = X_tensor.std(dim=0, keepdim=True)
std[std == 0] = 1.0
X_tensor = (X_tensor - mean) / std

test_X_tensor = (test_X_tensor - mean) / std

# guard against any remaining NaNs or infs after normalization
test_X_tensor = torch.nan_to_num(test_X_tensor, nan=0.0, posinf=0.0, neginf=0.0)

#Model
model = nn.Sequential(
    nn.Linear(10,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(),lr=0.01)

#Training
epochs = 5000
for epoch in range(epochs):
    model.train()
    
    logits = model(X_tensor)
    y_preds = torch.round(torch.sigmoid(logits))
    
    loss = loss_fn(logits, y_tensor)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    
with torch.inference_mode():
    model.eval()
    probs = torch.sigmoid(model(test_X_tensor))
    preds = torch.round(probs)
    # guard against NaNs/inf just in case and convert to boolean True/False for submission
    preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    prediction = preds.cpu().detach().numpy().astype(bool).flatten()

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': prediction
})
submission.to_csv("submission.csv", index=False)
print("Your submission was successfully saved!")


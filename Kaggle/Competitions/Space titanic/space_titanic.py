import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

BASE = Path(__file__).resolve().parent

train = pd.read_csv(BASE/"train.csv")
test = pd.read_csv(BASE/"test.csv")

#Cleaning
features = ["HomePlanet","CryoSleep","Destination","Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
train_x_df = train[features].copy()
train_y_df = train["Transported"].copy()

#Making categorical numeric data for features
train_x_df["VIP"] = train_x_df["VIP"].map({False:0, True:1, 'False':0, 'True':1})
print(train_x_df["HomePlanet"].unique())
train_x_df["CryoSleep"] = train_x_df["CryoSleep"].map({False:0, True:1, 'False':0, 'True':1})
train_x_df["HomePlanet"] = train_x_df["HomePlanet"].map({'Europa':0, 'Earth':1,'Mars':2})
train_x_df["Destination"] = train_x_df["Destination"].map({'TRAPPIST-1e':0, 'PSO J318.5-22':1,'55 Cancri e':2})
# print(train_x_df["HomePlanet"].isnull().sum())

#For targets 
train_y_df = train_y_df.map({False:0, True:1, 'False':0, 'True':1})

print(f"Shape of samples before removing NaN: {train_x_df.shape}")

# drop rows with any NaN in features and align targets
valid_idx = train_x_df.dropna().index
train_x_df = train_x_df.loc[valid_idx].reset_index(drop=True)
train_y_df = train_y_df.loc[valid_idx].reset_index(drop=True)

print(f"Shape of samples after removing NaN: {train_x_df.shape}")


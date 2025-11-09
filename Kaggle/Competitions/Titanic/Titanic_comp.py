import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

BASE = Path(__file__).resolve().parent

train = pd.read_csv(BASE/"train.csv")
test = pd.read_csv(BASE/"test.csv")
submission = pd.read_csv(BASE/"gender_submission.csv")

test.head()
train.head()

df = train.copy()
df_test = test.copy()

#Choose features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_x = df[features].copy()
train_y = df['Survived'].copy()
test_x =df_test[features].copy()
#Making all features numerical
train_x["Sex"] = train["Sex"].map({'male':0, 'female':1})
train_x['Embarked'] = train_x['Embarked'].fillna('S')  
test_x["Sex"] = test["Sex"].map({'male':0, 'female':1})
test_x['Embarked'] = test_x['Embarked'].fillna('S')  
# Fill missing numeric values 
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].median())
train_x['Fare'] = train_x['Fare'].fillna(train_x['Fare'].median())

test_x['Age'] = test_x['Age'].fillna(test_x['Age'].median())
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].median())

train_x = pd.get_dummies(train_x, columns=['Embarked'], drop_first=True)
test_x = pd.get_dummies(test_x, columns=['Embarked'], drop_first=True)

#Transforming into tensor
train_x = train_x.to_numpy(dtype=np.float32)
train_y = train_y.to_numpy(dtype=np.float32)
test_x = test_x.to_numpy(dtype=np.float32)


X = torch.tensor(train_x, dtype=torch.float32)
Y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
TEST = torch.tensor(test_x, dtype=torch.float32)

#normalize
mean = X.mean(dim=0)
std = X.std(dim=0)
std[std == 0] = 1.0
X = (X - mean) / std
TEST = (TEST - mean) / std

#Model
model = nn.Sequential(
    nn.Linear(8,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(),lr=0.003)
#Training
epochs = 10000
for epoch in range(epochs):
    model.train()
    
    logits = model(X)
    y_preds = torch.round(torch.sigmoid(logits))
    
    loss = loss_fn(logits, Y)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: {loss.item()}")


with torch.inference_mode():
    model.eval()
    probs = torch.sigmoid(model(TEST))
    preds = torch.round(probs)
    # guard against NaNs/inf just in case and convert to boolean True/False for submission
    preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    prediction = preds.cpu().detach().numpy().astype(bool).flatten()

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': prediction.astype(int)
})
submission.to_csv("submission.csv", index=False)
print("Your submission was successfully saved!")

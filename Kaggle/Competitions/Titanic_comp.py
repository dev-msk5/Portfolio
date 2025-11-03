import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test.head()
train.head()

df = train.copy()  # train loaded earlier from /kaggle/input/titanic/train.csv
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

#Model
def init_model():
    model = nn.Sequential(
        nn.Linear(8,15),
        nn.ReLU(),
        nn.Linear(15,3),
        nn.ReLU(),
        nn.Linear(3,1),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.BCEWithLogitsLoss()

    return model, optimizer, loss_function
def train_model(X, Y, epochs=500):
    model, optimizer, loss_function = init_model()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_function(outputs, Y)

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

    return model

model  = train_model(X, Y, 10000)

with torch.no_grad():
    # Perform a forward pass to get model predictions
    prediction = model(TEST)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': prediction.numpy().astype(int).flatten()
})
submission.to_csv("submission.csv", index=False)
print("Your submission was successfully saved!")

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam



def generate_content_data(n_users=10, n_items=20, feature_dim=10, seed=42):
    np.random.seed(seed)

    # Item feature vectors (movies described by 10 features)
    item_features = np.random.rand(n_items, feature_dim)  

    # User preference vectors (weights for 10 features)
    user_prefs = np.random.rand(n_users, feature_dim)  

    # Ratings = dot product of user preferences and item features
    ratings = user_prefs @ item_features.T  

    return item_features, user_prefs, ratings


# Example
items, users, ratings = generate_content_data()

print("Item features shape:", items.shape)    # (20, 10)
print("User preference shape:", users.shape)  # (5, 10)
print("Ratings shape:", ratings.shape)        # (5, 20)


# model
user_model = keras.Sequential([
    Dense(10,activation="relu"),
    Dense(12,activation="relu"),
    Dense(6,)
])

item_model = keras.Sequential([
    Dense(10,activation="relu"),
    Dense(12,activation="relu"),
    Dense(6,)
])

# create user input to NN
user_input = keras.layers.Input(shape=(users.shape[1]))
vu = user_model(user_input)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create item input to NN
item_input = keras.layers.Input(shape=(items.shape[1]))
vm = item_model(item_input)
vm = tf.linalg.l2_normalize(vm, axis=1)

# output
output = tf.keras.layers.Dot(axes=1)([vu,vm])
model = keras.Model([user_input,item_input],output)

model.compile(optimizer="adam",loss="mse")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- Option 1: XOR-like moons dataset ---
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# # --- Option 2: Concentric circles dataset ---
# X_circles, y_circles = make_circles(n_samples=500, factor=0.5, noise=0.1, random_state=42)

# # --- Option 3: Generic classification dataset ---
# X_class, y_class = make_classification(n_samples=500, n_features=2, 
#                                        n_informative=2, n_redundant=0,
#                                        n_clusters_per_class=1, random_state=42)

# Train + temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Split temp into validation + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)


# Plot
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Dataset for XGBoost Decision Tree")
plt.show()


# XGBoost model
model = XGBClassifier()

model.fit(X,y)
y_predict=model.predict(X_val)

# 5. Plot decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.show()

plot_decision_boundary(model, X_test, y_test, title="XGBoost Decision Boundary (Test Set)")
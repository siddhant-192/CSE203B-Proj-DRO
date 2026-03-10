import numpy as np
import cvxpy as cp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("--- EXPERIMENT 1: BREAST CANCER (L2 Wasserstein DRO) ---")
# 1. Data Setup (Labels must be -1 or 1 for SVM Hinge Loss)
data = load_breast_cancer()
X = data.data
y = np.where(data.target == 0, -1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

# 2. Global Corruption (Degraded sensors affect ALL features slightly)
np.random.seed(42)
X_test_corrupted = X_test + np.random.normal(0, 1.5, size=X_test.shape)

# 3. Base ERM Model (Standard Linear SVM)
beta_erm = cp.Variable(n_features)
# Hinge loss for classification
loss_erm = (1/n_samples) * cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ beta_erm)))
cp.Problem(cp.Minimize(loss_erm)).solve()

# 4. Standard L2 Wasserstein DRO (Dual is L2 Ridge Penalty)
beta_dro = cp.Variable(n_features)
loss_dro = (1/n_samples) * cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ beta_dro)))

lambda_radius = 0.5 # Ambiguity set radius
# L2 Norm Penalty (Unweighted)
dro_penalty = lambda_radius * cp.norm(beta_dro, 2) 
cp.Problem(cp.Minimize(loss_dro + dro_penalty)).solve()

# 5. Evaluation
def predict(X, beta): return np.sign(X @ beta.value)

print(f"ERM Accuracy (Clean):     {accuracy_score(y_test, predict(X_test, beta_erm)):.4f}")
print(f"ERM Accuracy (Corrupted): {accuracy_score(y_test, predict(X_test_corrupted, beta_erm)):.4f}")
print(f"DRO Accuracy (Clean):     {accuracy_score(y_test, predict(X_test, beta_dro)):.4f}")
print(f"DRO Accuracy (Corrupted): {accuracy_score(y_test, predict(X_test_corrupted, beta_dro)):.4f}\n")

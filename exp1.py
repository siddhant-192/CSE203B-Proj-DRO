import numpy as np
import cvxpy as cp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("--- EXPERIMENT 1: CALIFORNIA HOUSING ---")
data = fetch_california_housing()
X, y = data.data[:2000], data.target[:2000] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

noisy_indices = [2, 3]
np.random.seed(42)
X_test_corrupted = X_test.copy()
X_test_corrupted[:, noisy_indices] += np.random.normal(0, 3.0, size=(X_test.shape[0], len(noisy_indices)))

# 1. Base ERM
beta_erm = cp.Variable(n_features)
loss_erm = (1/n_samples) * cp.sum_squares(X_train @ beta_erm - y_train)
cp.Problem(cp.Minimize(loss_erm)).solve()

# 2. Standard W-DRO (Unweighted L-infinity penalty)
beta_std_dro = cp.Variable(n_features)
loss_std = (1/n_samples) * cp.sum_squares(X_train @ beta_std_dro - y_train)
lambda_radius = 0.5
std_penalty = lambda_radius * cp.max(cp.abs(beta_std_dro)) 
cp.Problem(cp.Minimize(loss_std + std_penalty)).solve()

# 3. Feature-Weighted W-DRO
beta_w_dro = cp.Variable(n_features)
loss_w = (1/n_samples) * cp.sum_squares(X_train @ beta_w_dro - y_train)
weights = np.ones(n_features)
weights[noisy_indices] = 0.1
w_penalty = lambda_radius * cp.max(cp.abs(beta_w_dro) / weights) 
cp.Problem(cp.Minimize(loss_w + w_penalty)).solve()

# Evaluation on both Clean and Corrupted Data
print("--- CLEAN DATA ---")
print(f"ERM MSE:           {mean_squared_error(y_test, X_test @ beta_erm.value):.4f}")
print(f"Standard DRO MSE:  {mean_squared_error(y_test, X_test @ beta_std_dro.value):.4f}")
print(f"Weighted DRO MSE:  {mean_squared_error(y_test, X_test @ beta_w_dro.value):.4f}")

print("\n--- CORRUPTED DATA ---")
print(f"ERM MSE:           {mean_squared_error(y_test, X_test_corrupted @ beta_erm.value):.4f}")
print(f"Standard DRO MSE:  {mean_squared_error(y_test, X_test_corrupted @ beta_std_dro.value):.4f}")
print(f"Weighted DRO MSE:  {mean_squared_error(y_test, X_test_corrupted @ beta_w_dro.value):.4f}\n")

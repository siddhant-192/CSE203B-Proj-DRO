import numpy as np
import cvxpy as cp
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("--- EXPERIMENT 2: DIABETES (Unweighted L1 Wasserstein DRO) ---")
# 1. Data Setup
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

# 2. Targeted Corruption (Corrupting features 2 and 3 heavily)
np.random.seed(42)
X_test_corrupted = X_test.copy()
X_test_corrupted[:, [2, 3]] += np.random.normal(0, 3.0, size=(X_test.shape[0], 2))

# 3. Base ERM Model (Standard Linear Regression)
beta_erm = cp.Variable(n_features)
loss_erm = (1/n_samples) * cp.sum_squares(X_train @ beta_erm - y_train)
cp.Problem(cp.Minimize(loss_erm)).solve()

# 4. Standard L1 Wasserstein DRO (Dual is Unweighted L-infinity Penalty)
beta_dro = cp.Variable(n_features)
loss_dro = (1/n_samples) * cp.sum_squares(X_train @ beta_dro - y_train)

lambda_radius = 20.0 # Ambiguity set radius
# L-infinity Norm Penalty (Unweighted)
dro_penalty = lambda_radius * cp.norm(beta_dro, "inf") 
cp.Problem(cp.Minimize(loss_dro + dro_penalty)).solve()

# 5. Evaluation
print(f"ERM MSE (Clean):     {mean_squared_error(y_test, X_test @ beta_erm.value):.1f}")
print(f"ERM MSE (Corrupted): {mean_squared_error(y_test, X_test_corrupted @ beta_erm.value):.1f}")
print(f"DRO MSE (Clean):     {mean_squared_error(y_test, X_test @ beta_dro.value):.1f}")
print(f"DRO MSE (Corrupted): {mean_squared_error(y_test, X_test_corrupted @ beta_dro.value):.1f}")

import numpy as np
import cvxpy as cp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("--- EXPERIMENT 2: BREAST CANCER (SVM Classification) ---")
data = load_breast_cancer()
X = data.data
y = np.where(data.target == 0, -1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

noisy_indices = [0, 1, 2, 3, 4]
np.random.seed(42)
X_test_corrupted = X_test.copy()
X_test_corrupted[:, noisy_indices] += np.random.normal(0, 2.0, size=(X_test.shape[0], len(noisy_indices)))

# 1. Base ERM
beta_erm = cp.Variable(n_features)
loss_erm = (1/n_samples) * cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ beta_erm)))
cp.Problem(cp.Minimize(loss_erm)).solve()

# 2. Standard W-DRO
beta_std_dro = cp.Variable(n_features)
loss_std = (1/n_samples) * cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ beta_std_dro)))
lambda_radius = 0.5
std_penalty = lambda_radius * cp.max(cp.abs(beta_std_dro)) 
cp.Problem(cp.Minimize(loss_std + std_penalty)).solve()

# 3. Feature-Weighted W-DRO
beta_w_dro = cp.Variable(n_features)
loss_w = (1/n_samples) * cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ beta_w_dro)))
weights = np.ones(n_features)
weights[noisy_indices] = 0.1
w_penalty = lambda_radius * cp.max(cp.abs(beta_w_dro) / weights) 
cp.Problem(cp.Minimize(loss_w + w_penalty)).solve()

def predict(X, beta): return np.sign(X @ beta.value)

print("--- CLEAN DATA ---")
print(f"ERM Accuracy:           {accuracy_score(y_test, predict(X_test, beta_erm)):.4f}")
print(f"Standard DRO Accuracy:  {accuracy_score(y_test, predict(X_test, beta_std_dro)):.4f}")
print(f"Weighted DRO Accuracy:  {accuracy_score(y_test, predict(X_test, beta_w_dro)):.4f}")

print("\n--- CORRUPTED DATA ---")
print(f"ERM Accuracy:           {accuracy_score(y_test, predict(X_test_corrupted, beta_erm)):.4f}")
print(f"Standard DRO Accuracy:  {accuracy_score(y_test, predict(X_test_corrupted, beta_std_dro)):.4f}")
print(f"Weighted DRO Accuracy:  {accuracy_score(y_test, predict(X_test_corrupted, beta_w_dro)):.4f}\n")

import numpy as np
import cvxpy as cp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. DATA SETUP & PREPROCESSING
# ==========================================
print("Loading California Housing Dataset...")
data = fetch_california_housing()

# Using a subset of 2000 samples to keep CVXPY solve times fast
X, y = data.data[:2000], data.target[:2000] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization is strictly required when using distance-based regularization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

# ==========================================
# 2. SIMULATE UNEVEN FEATURE NOISE (TEST DATA)
# ==========================================
# Simulating that Feature 2 (AveRooms) and Feature 3 (AveBedrms) are unreliable
noisy_feature_indices = [2, 3]
X_test_corrupted = X_test.copy()

# Inject heavy Gaussian noise only into the unreliable features
np.random.seed(42) # For reproducibility
noise_multiplier = 3.0
X_test_corrupted[:, noisy_feature_indices] += np.random.normal(
    0, noise_multiplier, size=(X_test.shape[0], len(noisy_feature_indices))
)

# ==========================================
# 3. BASE MODEL: Standard ERM (Linear Regression)
# ==========================================
print("\nTraining Base ERM Model...")
beta_erm = cp.Variable(n_features)

# Objective: Minimize standard Mean Squared Error
loss_erm = (1/n_samples) * cp.sum_squares(X_train @ beta_erm - y_train)
prob_erm = cp.Problem(cp.Minimize(loss_erm))
prob_erm.solve()

# ==========================================
# 4. PROPOSED MODEL: Feature-Weighted DRO
# ==========================================
print("Training Feature-Weighted DRO Model...")

# Define Weights (w_j): 1.0 means reliable/expensive to attack
weights = np.ones(n_features) 
# Tell DRO that features 2 and 3 are unreliable (cheap to attack)
weights[noisy_feature_indices] = 0.1 

beta_dro = cp.Variable(n_features)
loss_dro = (1/n_samples) * cp.sum_squares(X_train @ beta_dro - y_train)

# The Custom Weighted Penalty (Dual of the weighted L1 transport cost)
# Formula: lambda * max_j (|beta_j| / w_j)
lambda_radius = 0.5  # Size of the ambiguity set (hyperparameter)
dro_penalty = lambda_radius * cp.max(cp.abs(beta_dro) / weights)

prob_dro = cp.Problem(cp.Minimize(loss_dro + dro_penalty))
prob_dro.solve()

# ==========================================
# 5. EVALUATION & COMPARISON
# ==========================================
# Predictions for Base Model
y_pred_erm_clean = X_test @ beta_erm.value
y_pred_erm_corrupt = X_test_corrupted @ beta_erm.value

# Predictions for DRO Model
y_pred_dro_clean = X_test @ beta_dro.value
y_pred_dro_corrupt = X_test_corrupted @ beta_dro.value

print("\n" + "="*45)
print(" EXPERIMENT RESULTS ")
print("="*45)
print(f"Base Model MSE (Clean Test Data):     {mean_squared_error(y_test, y_pred_erm_clean):.4f}")
print(f"Base Model MSE (Corrupted Test Data): {mean_squared_error(y_test, y_pred_erm_corrupt):.4f}")
print("-" * 45)
print(f"DRO Model MSE (Clean Test Data):      {mean_squared_error(y_test, y_pred_dro_clean):.4f}")
print(f"DRO Model MSE (Corrupted Test Data):  {mean_squared_error(y_test, y_pred_dro_corrupt):.4f}")
print("="*45)

print("\n--- Coefficient Analysis ---")
print("Notice how the DRO model mathematically 'turns off' reliance on the noisy features (Indices 2 & 3):")
for i in range(n_features):
    status = "(Noisy)" if i in noisy_feature_indices else "       "
    print(f"Feature {i} {status}: Base Beta = {beta_erm.value[i]:>6.2f} | DRO Beta = {beta_dro.value[i]:>6.2f}")

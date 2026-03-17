import numpy as np
import cvxpy as cp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# IMPROVED BOOTSTRAP DRO WITH TWO FIXES:
#   1. Signal-to-noise ratio weights (instead of raw 1/variance)
#   2. Cross-validated lambda (instead of hardcoded 0.5)
# ==========================================

print("=" * 70)
print(" IMPROVED BOOTSTRAP WEIGHT ESTIMATION")
print("=" * 70)

# ==========================================
# 1. DATA SETUP
# ==========================================
print("\nLoading California Housing Dataset...")
data = fetch_california_housing()

X, y = data.data[:2000], data.target[:2000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples, n_features = X_train.shape

feature_names = data.feature_names

# ==========================================
# 2. CORRUPTION
# ==========================================
noisy_feature_indices = [2, 3]
X_test_corrupted = X_test.copy()
np.random.seed(42)
noise_multiplier = 3.0
X_test_corrupted[:, noisy_feature_indices] += np.random.normal(
    0, noise_multiplier, size=(X_test.shape[0], len(noisy_feature_indices))
)

# ==========================================
# 3. BOOTSTRAP ESTIMATION: TWO METHODS
# ==========================================
print("\nRunning bootstrap (B=200)...")

B = 200
bootstrap_betas = np.zeros((B, n_features))
rng = np.random.RandomState(42)

for b in range(B):
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_b, y_b = X_train[indices], y_train[indices]
    bootstrap_betas[b] = np.linalg.lstsq(X_b, y_b, rcond=None)[0]

beta_mean = np.mean(bootstrap_betas, axis=0)
beta_variance = np.var(bootstrap_betas, axis=0)

# --- Method A: Original (1/variance) ---
# Problem: large coefficients have large absolute variance even when stable.
# MedInc (beta~0.67, var~0.016) gets penalized despite being reliable.
eps = 1e-8
raw_inv_var = 1.0 / (beta_variance + eps)
weights_inv_var = raw_inv_var / np.mean(raw_inv_var)

# --- Method B: Signal-to-noise ratio (mean^2 / variance) ---
# Fixes the MedInc problem: large coefficient + moderate variance = high SNR.
# SNR = (E[beta_j])^2 / Var(beta_j)  (this is essentially t-statistic squared)
#
# MedInc: mean=0.67, var=0.016 -> SNR = 0.45/0.016 = 28.7  (high -> reliable)
# AveRooms: mean=0.35, var=0.30 -> SNR = 0.12/0.30 = 0.41  (low -> unreliable)
snr = beta_mean**2 / (beta_variance + eps)
weights_snr = snr / np.mean(snr)

print("\n--- Weight Comparison: Inverse Variance vs SNR ---")
print(f"{'Feature':<14} {'Mean β':<10} {'Var(β)':<12} {'1/Var wt':<10} {'SNR wt':<10}")
print("-" * 56)
for j in range(n_features):
    tag = " *" if j in noisy_feature_indices else "  "
    print(f"{feature_names[j]}{tag:<6} {beta_mean[j]:>+7.4f}  {beta_variance[j]:<12.6f} "
          f"{weights_inv_var[j]:<10.4f} {weights_snr[j]:<10.4f}")

print("\n* = corrupted at test time")
print("\nKey difference: MedInc (Feature 0)")
print(f"  1/Var weight: {weights_inv_var[0]:.4f}  (too low — kills best predictor)")
print(f"  SNR weight:   {weights_snr[0]:.4f}  (correctly identifies as reliable)")

# ==========================================
# 4. CROSS-VALIDATE LAMBDA
# ==========================================
print("\n--- Cross-Validating Lambda ---")

def train_weighted_dro(X_tr, y_tr, weights, lam):
    """Train weighted DRO and return coefficients."""
    n = X_tr.shape[0]
    beta = cp.Variable(X_tr.shape[1])
    loss = (1 / n) * cp.sum_squares(X_tr @ beta - y_tr)
    penalty = lam * cp.max(cp.abs(beta) / weights)
    prob = cp.Problem(cp.Minimize(loss + penalty))
    prob.solve(solver=cp.SCS, verbose=False)
    return beta.value

lambda_candidates = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# CV for each weight strategy
weight_strategies = {
    "Inverse Var": weights_inv_var,
    "SNR":         weights_snr,
}

best_lambdas = {}

for strategy_name, w in weight_strategies.items():
    print(f"\n  CV for {strategy_name} weights:")
    best_lam = None
    best_cv_mse = np.inf

    for lam in lambda_candidates:
        cv_mses = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            beta_cv = train_weighted_dro(X_tr, y_tr, w, lam)
            if beta_cv is not None:
                cv_mses.append(mean_squared_error(y_val, X_val @ beta_cv))

        if cv_mses:
            mean_cv = np.mean(cv_mses)
            marker = ""
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_lam = lam
                marker = " <-- best"
            print(f"    λ={lam:<5.2f}  CV MSE = {mean_cv:.4f}{marker}")

    best_lambdas[strategy_name] = best_lam
    print(f"  Best λ for {strategy_name}: {best_lam}")

# Also CV for oracle and isotropic
for strategy_name, w in [("Oracle", np.where(np.isin(range(n_features), noisy_feature_indices), 0.1, 1.0)),
                          ("Isotropic", np.ones(n_features))]:
    best_lam = None
    best_cv_mse = np.inf
    for lam in lambda_candidates:
        cv_mses = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            beta_cv = train_weighted_dro(X_tr, y_tr, w, lam)
            if beta_cv is not None:
                cv_mses.append(mean_squared_error(y_val, X_val @ beta_cv))
        if cv_mses:
            mean_cv = np.mean(cv_mses)
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_lam = lam
    best_lambdas[strategy_name] = best_lam

print(f"\n  Best λ for Oracle:    {best_lambdas['Oracle']}")
print(f"  Best λ for Isotropic: {best_lambdas['Isotropic']}")

# ==========================================
# 5. TRAIN ALL MODELS WITH BEST LAMBDAS
# ==========================================
print("\n--- Training Final Models with CV-Selected Lambda ---")

# ERM
beta_erm = cp.Variable(n_features)
loss = (1 / n_samples) * cp.sum_squares(X_train @ beta_erm - y_train)
cp.Problem(cp.Minimize(loss)).solve()

# Isotropic DRO
lam_iso = best_lambdas["Isotropic"]
beta_iso = cp.Variable(n_features)
loss = (1 / n_samples) * cp.sum_squares(X_train @ beta_iso - y_train)
penalty = lam_iso * cp.max(cp.abs(beta_iso))
cp.Problem(cp.Minimize(loss + penalty)).solve()

# Oracle
lam_oracle = best_lambdas["Oracle"]
oracle_weights = np.ones(n_features)
oracle_weights[noisy_feature_indices] = 0.1

beta_oracle = cp.Variable(n_features)
loss = (1 / n_samples) * cp.sum_squares(X_train @ beta_oracle - y_train)
penalty = lam_oracle * cp.max(cp.abs(beta_oracle) / oracle_weights)
cp.Problem(cp.Minimize(loss + penalty)).solve()

# Inverse Variance (original bootstrap)
lam_iv = best_lambdas["Inverse Var"]
beta_inv_var = cp.Variable(n_features)
loss = (1 / n_samples) * cp.sum_squares(X_train @ beta_inv_var - y_train)
penalty = lam_iv * cp.max(cp.abs(beta_inv_var) / weights_inv_var)
cp.Problem(cp.Minimize(loss + penalty)).solve()

# SNR (improved bootstrap)
lam_snr = best_lambdas["SNR"]
beta_snr = cp.Variable(n_features)
loss = (1 / n_samples) * cp.sum_squares(X_train @ beta_snr - y_train)
penalty = lam_snr * cp.max(cp.abs(beta_snr) / weights_snr)
cp.Problem(cp.Minimize(loss + penalty)).solve()

# ==========================================
# 6. RESULTS
# ==========================================
models = {
    "ERM":                          (beta_erm.value, "—"),
    "Isotropic DRO":                (beta_iso.value, lam_iso),
    "Oracle Weighted DRO":          (beta_oracle.value, lam_oracle),
    "Bootstrap (1/Var)":            (beta_inv_var.value, lam_iv),
    "Bootstrap (SNR)":              (beta_snr.value, lam_snr),
}

print("\n" + "=" * 75)
print(" FINAL RESULTS: IMPROVED BOOTSTRAP WITH CV LAMBDA")
print("=" * 75)
print(f"{'Model':<28} {'λ':<8} {'Clean MSE':<14} {'Corrupted MSE':<14} {'Degrad.':<12}")
print("-" * 75)

for name, (beta_val, lam) in models.items():
    mc = mean_squared_error(y_test, X_test @ beta_val)
    mr = mean_squared_error(y_test, X_test_corrupted @ beta_val)
    lam_str = f"{lam:<8.2f}" if isinstance(lam, float) else f"{'—':<8}"
    print(f"{name:<28} {lam_str} {mc:<14.4f} {mr:<14.4f} {mr-mc:<+12.4f}")

print("=" * 75)

# ==========================================
# 7. COEFFICIENT COMPARISON
# ==========================================
print("\n--- Coefficients ---")
print(f"{'Feature':<14} {'ERM':<10} {'Iso':<10} {'Oracle':<10} {'1/Var':<10} {'SNR':<10}")
print("-" * 64)
for j in range(n_features):
    tag = " *" if j in noisy_feature_indices else "  "
    print(f"{feature_names[j]}{tag:<6} "
          f"{beta_erm.value[j]:>8.4f}  "
          f"{beta_iso.value[j]:>8.4f}  "
          f"{beta_oracle.value[j]:>8.4f}  "
          f"{beta_inv_var.value[j]:>8.4f}  "
          f"{beta_snr.value[j]:>8.4f}")

# ==========================================
# 8. WEIGHT COMPARISON
# ==========================================
print("\n--- Weights ---")
print(f"{'Feature':<14} {'Oracle':<10} {'1/Var':<10} {'SNR':<10}")
print("-" * 44)
for j in range(n_features):
    tag = " *" if j in noisy_feature_indices else "  "
    print(f"{feature_names[j]}{tag:<6} "
          f"{oracle_weights[j]:<10.4f} "
          f"{weights_inv_var[j]:<10.4f} "
          f"{weights_snr[j]:<10.4f}")

# ==========================================
# 9. CORRUPTION SEVERITY SWEEP
# ==========================================
print("\n--- Corruption Severity Sweep ---")
print(f"{'σ':<6} {'ERM':<12} {'Iso DRO':<12} {'Oracle':<12} {'1/Var':<12} {'SNR':<12}")
print("-" * 66)

for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    X_sweep = X_test.copy()
    if sigma > 0:
        np.random.seed(42)
        X_sweep[:, noisy_feature_indices] += np.random.normal(
            0, sigma, size=(X_test.shape[0], len(noisy_feature_indices))
        )
    results = []
    for name, (beta_val, _) in models.items():
        results.append(mean_squared_error(y_test, X_sweep @ beta_val))

    print(f"{sigma:<6.1f} {results[0]:<12.4f} {results[1]:<12.4f} "
          f"{results[2]:<12.4f} {results[3]:<12.4f} {results[4]:<12.4f}")
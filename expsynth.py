import numpy as np
import cvxpy as cp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ==========================================
# SYNTHETIC VALIDATION: PROVING THE PIPELINE WORKS
# ==========================================
# In real data, we never know ground truth. Here we control everything:
#   - Which features are truly informative vs noise-only
#   - The exact corruption structure
#   - The true coefficients
#
# This lets us verify each claim:
#   1. Bootstrap SNR recovers the true noise structure
#   2. Weighted DRO suppresses the right features
#   3. Robustness under corruption follows
#
# DATA GENERATING PROCESS:
#   y = X * beta_true + epsilon
#
#   Features fall into three categories:
#     - Strong signals (large beta, low noise): should get HIGH weight
#     - Weak signals (small beta, high noise): should get LOW weight
#     - Pure noise (zero beta, any noise): should get LOW weight
#
#   At test time, we corrupt the weak/noisy features and verify that
#   SNR-weighted DRO is immune while ERM degrades.

print("=" * 75)
print(" SYNTHETIC VALIDATION EXPERIMENT")
print("=" * 75)

# ==========================================
# 1. DATA GENERATION
# ==========================================
np.random.seed(42)

n_train = 500
n_test = 500
d = 15  # total features

# True coefficient vector: 3 groups
# Group A (features 0-4):  Strong, reliable signals
# Group B (features 5-9):  Weak signals, will be corrupted
# Group C (features 10-14): Pure noise, zero true effect
beta_true = np.array([
    3.0, 2.5, 2.0, 1.5, 1.0,     # Group A: strong signals
    0.3, 0.2, 0.1, 0.15, 0.25,   # Group B: weak signals
    0.0, 0.0, 0.0, 0.0, 0.0,     # Group C: pure noise
])

group_A = list(range(0, 5))
group_B = list(range(5, 10))
group_C = list(range(10, 15))
noisy_features = group_B + group_C  # features 5-14 will be corrupted

# Generate features with some correlation within groups
def generate_X(n, d):
    """Generate features with within-group correlation."""
    X = np.random.randn(n, d)

    # Add within-group correlation (rho ~ 0.3) for Group A
    base_A = np.random.randn(n)
    for j in group_A:
        X[:, j] = 0.3 * base_A + np.sqrt(1 - 0.09) * X[:, j]

    # Add within-group correlation for Group B
    base_B = np.random.randn(n)
    for j in group_B:
        X[:, j] = 0.3 * base_B + np.sqrt(1 - 0.09) * X[:, j]

    # Group C: independent (pure noise features)
    return X

X_train = generate_X(n_train, d)
y_train = X_train @ beta_true + 0.5 * np.random.randn(n_train)

X_test = generate_X(n_test, d)
y_test = X_test @ beta_true + 0.5 * np.random.randn(n_test)

print(f"\nData: {n_train} train, {n_test} test, {d} features")
print(f"  Group A (features 0-4):   Strong signals, beta = {beta_true[group_A]}")
print(f"  Group B (features 5-9):   Weak signals,   beta = {beta_true[group_B]}")
print(f"  Group C (features 10-14): Pure noise,     beta = {beta_true[group_C]}")
print(f"  Corruption targets: Group B + C (features 5-14)")
print(f"  Label noise: σ = 0.5")

#==========================================
# 2. CORRUPTION
# ==========================================
noise_sigma = 3.0

X_test_corrupted = X_test.copy()
X_test_corrupted[:, noisy_features] += np.random.normal(
    0, noise_sigma, size=(n_test, len(noisy_features))
)

# Verify corruption hurts naive model
mse_clean = mean_squared_error(y_test, X_test @ beta_true)
mse_corrupt = mean_squared_error(y_test, X_test_corrupted @ beta_true)
print(f"\nTrue model MSE (clean):     {mse_clean:.4f}")
print(f"True model MSE (corrupted): {mse_corrupt:.4f}")
print(f"Degradation: +{mse_corrupt - mse_clean:.4f}")

# ==========================================
# 3. BOOTSTRAP WEIGHT ESTIMATION
# ==========================================
print("\n" + "=" * 75)
print(" STEP 1: DOES BOOTSTRAP SNR RECOVER THE NOISE STRUCTURE?")
print("=" * 75)

B = 300
bootstrap_betas = np.zeros((B, d))
rng = np.random.RandomState(42)

for b in range(B):
    idx = rng.choice(n_train, size=n_train, replace=True)
    X_b, y_b = X_train[idx], y_train[idx]
    bootstrap_betas[b] = np.linalg.lstsq(X_b, y_b, rcond=None)[0]

beta_mean = np.mean(bootstrap_betas, axis=0)
beta_var = np.var(bootstrap_betas, axis=0)

eps = 1e-8
snr = beta_mean**2 / (beta_var + eps)
weights_snr = snr / np.mean(snr)

raw_iv = 1.0 / (beta_var + eps)
weights_iv = raw_iv / np.mean(raw_iv)

# Oracle weights
oracle_weights = np.ones(d)
oracle_weights[noisy_features] = 0.1

print(f"\n{'Feature':<10} {'Group':<10} {'True β':<10} {'Mean β':<10} "
      f"{'Var(β)':<12} {'SNR wt':<10} {'Oracle wt':<10}")
print("-" * 72)
for j in range(d):
    if j in group_A:
        grp = "A (strong)"
    elif j in group_B:
        grp = "B (weak)"
    else:
        grp = "C (noise)"
    print(f"x{j:<8} {grp:<10} {beta_true[j]:>7.2f}   {beta_mean[j]:>+8.4f}  "
          f"{beta_var[j]:<12.6f} {weights_snr[j]:<10.4f} {oracle_weights[j]:<10.4f}")

# Check: do SNR weights correctly separate the groups?
mean_snr_A = np.mean(weights_snr[group_A])
mean_snr_B = np.mean(weights_snr[group_B])
mean_snr_C = np.mean(weights_snr[group_C])

print(f"\nMean SNR weight by group:")
print(f"  Group A (strong signals): {mean_snr_A:.4f}  (should be HIGH)")
print(f"  Group B (weak signals):   {mean_snr_B:.4f}  (should be LOW)")
print(f"  Group C (pure noise):     {mean_snr_C:.4f}  (should be LOWEST)")

if mean_snr_A > mean_snr_B > mean_snr_C:
    print("  -> CORRECT RANKING: A > B > C")
elif mean_snr_A > max(mean_snr_B, mean_snr_C):
    print("  -> PARTIALLY CORRECT: A is highest, B/C ordering may be reversed")
else:
    print("  -> UNEXPECTED RANKING: investigate bootstrap estimates")

# ==========================================
# 4. CROSS-VALIDATE LAMBDA
# ==========================================
print("\n" + "=" * 75)
print(" STEP 2: CROSS-VALIDATE LAMBDA FOR EACH STRATEGY")
print("=" * 75)

def train_dro(X_tr, y_tr, weights, lam):
    n, p = X_tr.shape
    beta = cp.Variable(p)
    loss = (1 / n) * cp.sum_squares(X_tr @ beta - y_tr)
    penalty = lam * cp.max(cp.abs(beta) / weights)
    prob = cp.Problem(cp.Minimize(loss + penalty))
    prob.solve(solver=cp.SCS, verbose=False)
    return beta.value

lambda_candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

strategies = {
    "Isotropic": np.ones(d),
    "Oracle":    oracle_weights,
    "1/Var":     weights_iv,
    "SNR":       weights_snr,
}

best_lambdas = {}

for sname, w in strategies.items():
    best_lam, best_mse = None, np.inf
    for lam in lambda_candidates:
        mses = []
        for tr_idx, val_idx in kf.split(X_train):
            bv = train_dro(X_train[tr_idx], y_train[tr_idx], w, lam)
            if bv is not None:
                mses.append(mean_squared_error(y_train[val_idx], X_train[val_idx] @ bv))
        if mses:
            m = np.mean(mses)
            if m < best_mse:
                best_mse = m
                best_lam = lam
    best_lambdas[sname] = best_lam
    print(f"  {sname:<12} best λ = {best_lam}, CV MSE = {best_mse:.4f}")

# ==========================================
# 5. TRAIN FINAL MODELS
# ==========================================
print("\nTraining final models...")

# ERM
beta_erm = cp.Variable(d)
loss = (1 / n_train) * cp.sum_squares(X_train @ beta_erm - y_train)
cp.Problem(cp.Minimize(loss)).solve()

final_betas = {"ERM": beta_erm.value}
for sname, w in strategies.items():
    bv = train_dro(X_train, y_train, w, best_lambdas[sname])
    final_betas[sname] = bv

# ==========================================
# 6. RESULTS
# ==========================================
print("\n" + "=" * 75)
print(" STEP 3: DOES WEIGHTED DRO SUPPRESS THE RIGHT FEATURES?")
print("=" * 75)

print(f"\n{'Feature':<10} {'Group':<10} {'True β':<8} {'ERM':<10} {'Iso':<10} "
      f"{'Oracle':<10} {'SNR':<10}")
print("-" * 68)
for j in range(d):
    if j in group_A:
        grp = "A"
    elif j in group_B:
        grp = "B"
    else:
        grp = "C"
print(f"x{j:<8} {grp:<10} {beta_true[j]:>6.2f}  "
        f"{final_betas['ERM'][j]:>+8.4f}  "
        f"{final_betas['Isotropic'][j]:>+8.4f}  "
        f"{final_betas['Oracle'][j]:>+8.4f}  "
        f"{final_betas['SNR'][j]:>+8.4f}")

# Compute how well each method preserves Group A and suppresses B+C
print("\n--- Coefficient Fidelity ---")
for name in ["ERM", "Isotropic", "Oracle", "SNR"]:
    bv = final_betas[name]
    # Group A: how close to true? (lower = better)
    err_A = np.sqrt(np.mean((bv[group_A] - beta_true[group_A])**2))
    # Group B+C: how close to zero? (lower = better)
    err_BC = np.sqrt(np.mean(bv[noisy_features]**2))
    print(f"  {name:<12}  Group A RMSE from true: {err_A:.4f}  |  "
          f"Group B+C RMS magnitude: {err_BC:.4f}")

# ==========================================
# 7. ROBUSTNESS EVALUATION
# ==========================================
print("\n" + "=" * 75)
print(" STEP 4: ROBUSTNESS UNDER CORRUPTION")
print("=" * 75)

print(f"\n{'Model':<20} {'λ':<8} {'Clean MSE':<14} {'Corrupt MSE':<14} {'Degrad.':<12}")
print("-" * 68)
for name, bv in final_betas.items():
    mc = mean_squared_error(y_test, X_test @bv)
    mr = mean_squared_error(y_test, X_test_corrupted @ bv)
    lam_str = f"{best_lambdas.get(name, 0):.2f}" if name != "ERM" else "—"
    print(f"{name:<20} {lam_str:<8} {mc:<14.4f} {mr:<14.4f} {mr-mc:<+12.4f}")

# ==========================================
# 8. CORRUPTION SEVERITY SWEEP
# ==========================================
print("\n--- Corruption Severity Sweep ---")
print(f"{'σ':<6} {'ERM':<12} {'Iso':<12} {'Oracle':<12} {'1/Var':<12} {'SNR':<12}")
print("-" * 66)

for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    X_sw = X_test.copy()
    if sigma > 0:
        np.random.seed(42)
        X_sw[:, noisy_features] += np.random.normal(
            0, sigma, size=(n_test, len(noisy_features))
        )
    row = f"{sigma:<6.1f}"
    for name in ["ERM", "Isotropic", "Oracle", "1/Var", "SNR"]:
        mse = mean_squared_error(y_test, X_sw @ final_betas[name])
        row += f" {mse:<12.4f}"
    print(row)

# ==========================================
# 9. STATISTICAL VALIDATION (MULTIPLE SEEDS)
# ==========================================
# Run the full pipeline on multiple random seeds to verify the result
# isn't an artifact of one particular data split.

print("\n" + "=" * 75)
print(" STEP 5: STABILITY ACROSS RANDOM SEEDS")
print("=" * 75)

n_seeds = 10
results_by_seed = {name: {"clean": [], "corrupt": []} for name in
                   ["ERM", "Oracle", "SNR"]}
snr_ranking_correct = 0

for seed in range(n_seeds):
    rng_s = np.random.RandomState(seed * 100 + 7)

    # Generate fresh data
    X_tr_s = generate_X(n_train, d)
    y_tr_s = X_tr_s @ beta_true + 0.5 * rng_s.randn(n_train)
    X_te_s = generate_X(n_test, d)
    y_te_s = X_te_s @ beta_true + 0.5 * rng_s.randn(n_test)

    X_te_corrupt_s = X_te_s.copy()
    X_te_corrupt_s[:, noisy_features] += rng_s.normal(0, noise_sigma,
                                                       size=(n_test, len(noisy_features)))

    # Bootstrap SNR on this seed's training data
    bb = np.zeros((B, d))
    for b in range(B):
        idx = rng_s.choice(n_train, size=n_train, replace=True)
        bb[b] = np.linalg.lstsq(X_tr_s[idx], y_tr_s[idx], rcond=None)[0]

    bm = np.mean(bb, axis=0)
    bv_var = np.var(bb, axis=0)
    s = bm**2 / (bv_var + eps)
    w_s = s / np.mean(s)

    # Check ranking
    if np.mean(w_s[group_A]) > np.mean(w_s[group_B]) > np.mean(w_s[group_C]):
        snr_ranking_correct += 1

    # Train ERM
    b_erm = cp.Variable(d)
    cp.Problem(cp.Minimize(
        (1/n_train) * cp.sum_squares(X_tr_s @ b_erm - y_tr_s)
    )).solve(solver=cp.SCS, verbose=False)

    # Train Oracle DRO (use previously found lambda)
    b_ora = train_dro(X_tr_s, y_tr_s, oracle_weights, best_lambdas["Oracle"])

    # Train SNR DRO
    b_snr = train_dro(X_tr_s, y_tr_s, w_s, best_lambdas["SNR"])

    for name, bval in [("ERM", b_erm.value), ("Oracle", b_ora), ("SNR", b_snr)]:
        if bval is not None:
            results_by_seed[name]["clean"].append(
                mean_squared_error(y_te_s, X_te_s @ bval))
            results_by_seed[name]["corrupt"].append(
                mean_squared_error(y_te_s, X_te_corrupt_s @ bval))

print(f"\nResults across {n_seeds} random seeds:")
print(f"{'Model':<12} {'Clean MSE':<20} {'Corrupt MSE':<20}")
print(f"{'':12} {'mean ± std':<20} {'mean ± std':<20}")
print("-" * 52)
for name in ["ERM", "Oracle", "SNR"]:
    cl = results_by_seed[name]["clean"]
    cr = results_by_seed[name]["corrupt"]
    if cl:
        print(f"{name:<12} {np.mean(cl):.4f} ± {np.std(cl):.4f}      "
              f"{np.mean(cr):.4f} ± {np.std(cr):.4f}")

print(f"\nSNR weight ranking correct (A > B > C): {snr_ranking_correct}/{n_seeds} seeds")

# ==========================================
# 10. SUMMARY
# ========================================
print("\n" + "=" * 75)
print(" SUMMARY OF CLAIMS")
print("=" * 75)
print("""
Claim 1: Bootstrap SNR recovers the noise structure.
  -> Check the weight table (Step 1). Group A should have high weights,
     Group C should have near-zero weights, Group B in between.
  -> The multi-seed ranking test (Step 5) shows how reliably this holds.

Claim 2: SNR-weighted DRO suppresses the right features.
  -> Check the coefficient table (Step 3). Group A betas should be close
     to their true values. Group B+C betas should be near zero.
  -> The "Coefficient Fidelity" metric quantifies this.

Claim 3: This translates to robustness.
  -> Check the severity sweep (Step 4). SNR line should be nearly flat
     while ERM explodes.
  -> The multi-seed test (Step 5) confirms this isn't a lucky split.

This synthetic experiment validates the pipeline BEFORE applying it to
real data (California Housing, Breast Cancer, Diabetes), where ground
truth is unknown.
""")

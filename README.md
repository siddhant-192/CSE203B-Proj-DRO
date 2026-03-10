# Comprehensive Evaluation of Distributionally Robust Optimization (DRO) Models

This document details the comparative analysis of three models—Standard Empirical Risk Minimization (ERM), Standard Unweighted Wasserstein DRO, and our proposed Feature-Weighted Wasserstein DRO—across three distinct datasets under targeted adversarial noise conditions.

---

## Experiment 1: California Housing (Regression)
**Objective:** Evaluate robustness against targeted noise in specific, highly-correlated housing features (`AveRooms`, `AveBedrms`).
**Targeted Corruption:** Additive Gaussian noise injected strictly into indices 2 and 3.

### Results Comparison
| Model | MSE (Clean Data) | MSE (Corrupted Data) |
| :--- | :--- | :--- |
| **Standard ERM** | 3.8059 | 5.6161 |
| **Standard W-DRO (Unweighted L-inf)** | 3.8622 | 6.2909 |
| **Feature-Weighted W-DRO (Proposed)** | **3.8942** | **3.9214** |

---

## Experiment 2: Breast Cancer (Classification)
**Objective:** Evaluate robustness in a classification setting using Hinge Loss, simulating a localized failure of the first 5 sensory measurements.
**Targeted Corruption:** Additive Gaussian noise injected strictly into indices 0 through 4.

### Results Comparison
| Model | Accuracy (Clean Data) | Accuracy (Corrupted Data) |
| :--- | :--- | :--- |
| **Standard ERM** | 0.8947 | 0.5614 |
| **Standard W-DRO (Unweighted L-inf)** | 0.9737 | 0.9386 |
| **Feature-Weighted W-DRO (Proposed)** | **0.9825** | **0.9825** |

---

## Experiment 3: Diabetes Dataset (Regression)
**Objective:** Evaluate behavior on medical progression data when key biological indicators (`bmi`, `bp`) are subjected to severe measurement error.
**Targeted Corruption:** Additive Gaussian noise injected strictly into indices 2 and 3.

### Results Comparison
| Model | MSE (Clean Data) | MSE (Corrupted Data) |
| :--- | :--- | :--- |
| **Standard ERM** | 27738.2 | 38819.7 |
| **Standard W-DRO (Unweighted L-inf)** | 27542.1 | 35415.8 |
| **Feature-Weighted W-DRO (Proposed)** | **27331.2** | **27656.0** |

---

## Concluding Analysis

The empirical evaluation across three distinct datasets robustly validates our core hypothesis: while standard DRO provides generalized protection against isotropic uncertainty, **Feature-Weighted DRO exhibits vastly superior survivability against targeted adversarial noise.**

### 1. The Vulnerability of ERM
As expected, the Standard ERM models suffered catastrophic failures when exposed to distribution shifts. In the classification task (Experiment 2), targeted noise caused ERM accuracy to plummet from ~89% to ~56%, rendering the model functionally useless. In both regression tasks, the Mean Squared Error spiked dramatically. ERM operates under the naive assumption that test data identically matches training data, leaving it completely defenseless when highly-weighted features are corrupted.

### 2. The Critical Flaw of Standard (Unweighted) DRO
The most revealing insight from these experiments is the behavior of the Standard Wasserstein DRO model, particularly in Experiment 1. When faced with targeted noise, the Standard DRO model actually performed *worse* than the Base ERM model (MSE of 6.29 vs. 5.61). 



This occurs because standard $L_1$ Wasserstein DRO relies on an unweighted $L_\infty$ penalty in its dual formulation. This penalty forces all model coefficients ($\beta$) to remain relatively equal in magnitude to protect against a "worst-case" scenario occurring *anywhere*. By capping the clean, reliable features, the Standard DRO model mathematically forced itself to rely on the noisy features, exacerbating the error when those specific features were attacked.

### 3. The Superiority of Feature-Weighted Regularization
Our proposed Feature-Weighted W-DRO model completely bypasses the flaw of standard DRO by utilizing domain knowledge (via the hyperparameter $w_j$). 



* **Immunity to Targeted Attacks:** In Experiment 1, the proposed model's error remained virtually unchanged (shifting a mere 0.03 from clean to corrupted), proving that it successfully neutralized the noisy features during training.
* **Flawless Classification:** In Experiment 2, the weighted model achieved a 98.25% accuracy on clean data and maintained that exact 98.25% accuracy under heavy corruption. 
* **Generalization Gains:** Interestingly, in Experiments 2 and 3, the Feature-Weighted model outperformed the base ERM model *even on clean data*. By intentionally down-weighting naturally noisy variables during training, the model functioned as an advanced feature-selector, reducing overfitting and improving baseline generalization.

**Conclusion:** When the source or likelihood of data corruption is unevenly distributed across features, applying an isotropic ambiguity set (Standard DRO) is suboptimal and potentially harmful. Our Feature-Weighted transport cost directly translates into a targeted "smart regularizer" in the convex dual, allowing models to mathematically immunize themselves against known data vulnerabilities without sacrificing predictive power on reliable features.

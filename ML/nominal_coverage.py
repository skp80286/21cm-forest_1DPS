import pandas as pd
import numpy as np
import sys 
from scipy.stats import kstest
import matplotlib.pyplot as plt

# ------------------------
# Load and prepare data
# ------------------------
df = pd.read_csv(sys.argv[1])

# Rename columns for clarity
df.rename(columns={
    "pred_xHI": "predicted_xHI",
    "pred_logfX": "predicted_logfx",
    "test_xHI": "true_xHI",
    "test_logfX": "true_logfx"
}, inplace=True)

# Group by true parameter pairs
groups = df.groupby(["true_xHI", "true_logfx"])

results = []
coverages = {0.5: [], 0.68: [], 0.9: [], 0.95: []}

# ------------------------
# PIT and nominal coverage
# ------------------------
for (true_xHI, true_logfx), g in groups:
    true_xHI_val = true_xHI
    true_logfx_val = true_logfx

    preds_xHI = np.sort(g["predicted_xHI"].values)
    preds_logfx = np.sort(g["predicted_logfx"].values)
    n = len(preds_xHI)

    # PIT (percentile) values
    u_xHI = np.searchsorted(preds_xHI, true_xHI_val, side="right") / n
    u_logfx = np.searchsorted(preds_logfx, true_logfx_val, side="right") / n

    results.append({
        "true_xHI": true_xHI_val,
        "true_logfx": true_logfx_val,
        "u_xHI": u_xHI,
        "u_logfx": u_logfx
    })

    # Nominal coverage check
    for alpha in coverages.keys():
        lower = (1 - alpha) / 2
        upper = 1 - lower
        q_low_xHI, q_high_xHI = np.quantile(preds_xHI, [lower, upper])
        q_low_logfx, q_high_logfx = np.quantile(preds_logfx, [lower, upper])

        I_xHI = int(q_low_xHI <= true_xHI_val <= q_high_xHI)
        I_logfx = int(q_low_logfx <= true_logfx_val <= q_high_logfx)

        coverages[alpha].append((I_xHI, I_logfx))

# PIT results DataFrame
pit_df = pd.DataFrame(results)

# ------------------------
# KS uniformity test
# ------------------------
ks_xHI = kstest(pit_df["u_xHI"], "uniform", args=(0, 1))
ks_logfx = kstest(pit_df["u_logfx"], "uniform", args=(0, 1))

print("Kolmogorov–Smirnov Test Results:")
print(f"xHI    -> KS statistic = {ks_xHI.statistic:.3f}, p-value = {ks_xHI.pvalue:.3f}")
print(f"log fX -> KS statistic = {ks_logfx.statistic:.3f}, p-value = {ks_logfx.pvalue:.3f}")

# ------------------------
# Nominal coverage results
# ------------------------
nominal_results = []
for alpha, indicators in coverages.items():
    xHI_vals = [i[0] for i in indicators]
    logfx_vals = [i[1] for i in indicators]

    emp_cov_xHI = np.mean(xHI_vals)
    emp_cov_logfx = np.mean(logfx_vals)

    se_xHI = np.sqrt(emp_cov_xHI * (1 - emp_cov_xHI) / len(xHI_vals))
    se_logfx = np.sqrt(emp_cov_logfx * (1 - emp_cov_logfx) / len(logfx_vals))

    nominal_results.append({
        "alpha": alpha,
        "emp_cov_xHI": emp_cov_xHI,
        "se_xHI": se_xHI,
        "emp_cov_logfx": emp_cov_logfx,
        "se_logfx": se_logfx
    })

nominal_results_df = pd.DataFrame(nominal_results)

print("\nNominal Coverage Results:")
print(nominal_results_df)

# ------------------------
# Plots
# ------------------------

# PIT histograms
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(pit_df["u_xHI"], bins=10, range=(0, 1), edgecolor='black')
ax[0].set_title("PIT Histogram for xHI")
ax[0].set_xlabel("u_xHI")
ax[0].set_ylabel("Count")

ax[1].hist(pit_df["u_logfx"], bins=10, range=(0, 1), edgecolor='black')
ax[1].set_title("PIT Histogram for log fX")
ax[1].set_xlabel("u_logfx")

plt.tight_layout()
plt.show()

# Calibration curve plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# xHI plot
ax[0].errorbar(
    nominal_results_df["alpha"],
    nominal_results_df["emp_cov_xHI"],
    yerr=nominal_results_df["se_xHI"],
    fmt="o-",
    capsize=4,
    label="Empirical"
)
ax[0].plot([0, 1], [0, 1], "k--", label="Ideal calibration")
ax[0].set_xlabel("Nominal coverage α")
ax[0].set_ylabel("Empirical coverage")
ax[0].set_title("Nominal Coverage – xHI")
ax[0].set_xlim(0.4, 1.0)
ax[0].set_ylim(0, 1.05)
ax[0].legend()

# log fX plot
ax[1].errorbar(
    nominal_results_df["alpha"],
    nominal_results_df["emp_cov_logfx"],
    yerr=nominal_results_df["se_logfx"],
    fmt="o-",
    capsize=4,
    label="Empirical"
)
ax[1].plot([0, 1], [0, 1], "k--", label="Ideal calibration")
ax[1].set_xlabel("Nominal coverage α")
ax[1].set_title("Nominal Coverage – log fX")
ax[1].set_xlim(0.4, 1.0)
ax[1].set_ylim(0, 1.05)
ax[1].legend()

plt.tight_layout()
plt.show()


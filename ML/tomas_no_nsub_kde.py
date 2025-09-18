import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# -------------------------------------------------------------------
# 1.  File names and the associated test (x, y) points
# -------------------------------------------------------------------
files_and_tests = [
    ("tomas_data/21cm-forest_1DPS/MCMC_samples/flatsamp_200Mpc_xHI0.11_fX-1.0_uGMRT_8kHz_Smin64.2mJy_alphaR-0.44_t50h_dk0.25_6kbins_100000steps.npy",
     (0.11, -1.0)),
    ("tomas_data/21cm-forest_1DPS/MCMC_samples/flatsamp_200Mpc_xHI0.11_fX-3.0_uGMRT_8kHz_Smin64.2mJy_alphaR-0.44_t50h_dk0.25_6kbins_100000steps.npy",
     (0.11, -3.0)),
    ("tomas_data/21cm-forest_1DPS/MCMC_samples/flatsamp_200Mpc_xHI0.52_fX-2.0_uGMRT_8kHz_Smin64.2mJy_alphaR-0.44_t50h_dk0.25_6kbins_100000steps.npy",
     (0.52, -2.0)),
    ("tomas_data/21cm-forest_1DPS/MCMC_samples/flatsamp_200Mpc_xHI0.80_fX-1.0_uGMRT_8kHz_Smin64.2mJy_alphaR-0.44_t50h_dk0.25_6kbins_100000steps.npy",
     (0.80, -1.0)),
    ("tomas_data/21cm-forest_1DPS/MCMC_samples/flatsamp_200Mpc_xHI0.80_fX-3.0_uGMRT_8kHz_Smin64.2mJy_alphaR-0.44_t50h_dk0.25_6kbins_100000steps.npy",
     (0.80, -3.0)),
]

# -------------------------------------------------------------------
# 2.  Plot set-up and a common grid for KDE evaluation
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

all_x, all_y = [], []
for fname, _ in files_and_tests:
    arr = np.load(fname)
    all_x.append(arr[1000:, 0])
    all_y.append(arr[1000:, 1])

# use global limits for a consistent contour grid
x_min, x_max = np.min(np.hstack(all_x)) - .1, np.max(np.hstack(all_x)) + .1
y_min, y_max = np.min(np.hstack(all_y)) - .1, np.max(np.hstack(all_y)) + .1
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
grid_pos = np.vstack([XX.ravel(), YY.ravel()])

# -------------------------------------------------------------------
# 3.  Loop over the five samples
# -------------------------------------------------------------------
tile_area = 0.05 * 0.2
prob_each, prod_prob = [], 1.0

for (fname, (tx, ty)), xs, ys in zip(files_and_tests, all_x, all_y):
    # scatter
    #ax.scatter(xs, ys, s=5, alpha=0.3, label=f"{fname[:28]}…")

    # KDE
    kde = gaussian_kde(np.vstack([xs, ys]))
    ZZ = kde(grid_pos).reshape(XX.shape)
    ax.contour(XX, YY, ZZ, levels=6, linewidths=1, alpha=.8)

    # star at the test point
    ax.plot(tx, ty, marker="*", markersize=12, color="black")

    # probability for this tile
    p = kde([tx, ty])[0] * tile_area
    prob_each.append(p)
    prod_prob *= p

# -------------------------------------------------------------------
# 4.  Plot cosmetics
# -------------------------------------------------------------------
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Five samples (from .npy files) with Gaussian-KDE contours")
ax.legend(fontsize="x-small", frameon=False)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 5.  Report numbers
# -------------------------------------------------------------------
print("Probability in the Δx=0.05 × Δy=0.2 tile centred on each test value:")
for i, (prob, (_, (tx, ty))) in enumerate(zip(prob_each, files_and_tests), 1):
    print(f"  Sample {i} at ({tx:.2f}, {ty:.1f}):  {prob:.6e}")
print(f"\nJoint probability (product of the five): {prod_prob:.6e}")
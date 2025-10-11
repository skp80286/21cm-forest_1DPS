import pandas as pd
import numpy as np
from scipy.stats import kstest

# Load the data
df = pd.read_csv("saved_output/inference_gmrt50h/latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250709134119/test_results.csv")
#df = pd.read_csv("saved_output/inference_gmrt50h/latentdiffseed_f21_inference_latent_train_test_uGMRT_t50.0_20250822113133/test_results.csv")

# Group data by each true parameter pair
groups = df.groupby(["test_xHI", "test_logfX"])

results = {}
"""
# Loop over each test case (5 groups total)
for i, ((test_xHI, test_logfX), g) in enumerate(groups, 1):
    results[i] = {'test_xHI':test_xHI, 'test_logfX':test_logfX}
    print(f'test_xHI:{test_xHI:.2f}, test_logfX:{test_logfX:.2f}, g:{g.shape}\n{g.head(1)}\n{g.tail(1)}')
    # --- xHI ---
    preds_xHI = np.sort(g["pred_xHI"].values)
    # percentile of test_xHI within posterior
    u_xHI = np.searchsorted(preds_xHI, test_xHI, side="right") / len(preds_xHI)
    
    # KS test against Uniform[0,1]
    ks_stat_xHI, pval_xHI = kstest([u_xHI], 'uniform', args=(0, 1), N=len(preds_xHI))
    results[i]['xHI'] = (ks_stat_xHI, pval_xHI)
    
    # --- logfx ---
    preds_logfx = np.sort(g["pred_logfX"].values)
    u_logfx = np.searchsorted(preds_logfx, test_logfX, side="right") / len(preds_logfx)
    
    # KS test against Uniform[-4, 1] (shift uniform to [0,1])
    u_scaled = (u_logfx - (-4)) / (1 - (-4))  # scaling to [0,1]
    ks_stat_logfx, pval_logfx = kstest([u_scaled], 'uniform', args=(0, 1))
    results[i]['logfx'] = (ks_stat_logfx, pval_logfx)
"""
# --- Combine all 5 test cases together ---
all_u_xHI = []
all_u_logfx = []

for (test_xHI, test_logfX), g in groups:
    #if f'test_xHI:{test_xHI:.2f}, test_logfX:{test_logfX:.2f}' == 'test_xHI:0.11, test_logfX:-1.00':
    #    continue
    #print(f'test_xHI:{test_xHI:.2f}, test_logfX:{test_logfX:.2f}, g:{g.shape}\n{g.head(1)}\n{g.tail(1)}')
    preds_xHI = np.sort(g["pred_xHI"].values)
    u_xHI = np.searchsorted(preds_xHI, test_xHI, side="right") / len(preds_xHI)
    all_u_xHI.append(u_xHI)

    preds_logfx = np.sort(g["pred_logfX"].values)
    u_logfx = np.searchsorted(preds_logfx, test_logfX, side="right") / len(preds_logfx)
    #u_scaled = (u_logfx - (-4)) / (1 - (-4))
    #print(f'u_logfx={u_logfx}, u_scaled={u_scaled}')
    all_u_logfx.append(u_logfx)
    print(f'test_xHI:{test_xHI:.2f}, test_logfX:{test_logfX:.2f}, u_xHI:{u_xHI}, u_logfX={u_logfx}')

ks_all_xHI, p_all_xHI = kstest(all_u_xHI, 'uniform', args=(0, 1), N =10000)
ks_all_logfx, p_all_logfx = kstest(all_u_logfx, 'uniform', args=(0, 1), N =10000)

# --- Print Results ---
for i in results:
    print(f"Test case {i}:")
    print(f"  xHI={results[i]['test_xHI']:.2f}  -> KS={results[i]['xHI'][0]:.4f}, p={results[i]['xHI'][1]:.4f}")
    print(f"  logfX={results[i]['test_logfX']:.2f}  -> KS={results[i]['logfx'][0]:.4f}, p={results[i]['logfx'][1]:.4f}")

print("\nCombined over all 5 test cases:")
print(f"  xHI   -> KS={ks_all_xHI:.4f}, p={p_all_xHI:.4f}")
print(f"  logfx -> KS={ks_all_logfx:.4f}, p={p_all_logfx:.4f}")

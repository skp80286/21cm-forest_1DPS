import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import f21_predict_base as base

# -------------------------------------------------------
# 1. Load XGBoost model
# -------------------------------------------------------

def load_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".model", ".ubj"]:
        booster = xgb.Booster()
        booster.load_model(path)
        return booster, True
    else:
        model = joblib.load(path)
        return model, False


class ModelWrapper:
    def __init__(self, model, is_booster=False):
        self.model = model
        self.is_booster = is_booster

    def predict(self, X):
        if self.is_booster:
            dmat = xgb.DMatrix(X)
            out = self.model.predict(dmat)
        else:
            out = self.model.predict(X)
        return np.asarray(out)[:, :2]


# -------------------------------------------------------
# 2. Load grid of npy latent feature files
# -------------------------------------------------------

FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_filename(path):
    base = os.path.basename(path)
    m = re.search(rf"xHI({FLOAT_RE})_logfX({FLOAT_RE})", base)
    if m is None:
        raise ValueError(f"Cannot parse parameters from {base}")
    return float(m.group(1)), float(m.group(2))

def load_feature_grid(features_dir):
    files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    grid = []
    for f in files:
        try:
            xHI, logfX = parse_filename(f)
        except:
            continue
        arr = np.load(os.path.join(features_dir, f))
        grid.append((xHI, logfX, arr))
    return grid


# -------------------------------------------------------
# 3. Feature interpolation
# -------------------------------------------------------

def k_nearest(xHI, logfX, grid, k=4):
    d = []
    for (xh, lf, arr) in grid:
        dist = math.hypot(xHI - xh, logfX - lf)
        d.append(dist)
    idx = np.argsort(d)[:k]
    return [grid[i] for i in idx], np.array(d)[idx]


def interpolate_features(xHI, logfX, grid, n_samples=2000, k=4, samples_per_neighbor=1):
    neigh, d = k_nearest(xHI, logfX, grid, k)

    if np.any(d == 0):
        idx = int(np.argmin(d))
        arr = neigh[idx][2]
        idxs = np.random.randint(0, arr.shape[0], size=n_samples)
        return arr[idxs]

    eps = 1e-8
    w = 1.0 / (d + eps)
    w /= w.sum()

    n_features = neigh[0][2].shape[1]
    out = np.zeros((n_samples, n_features))

    for weight, (xh, lf, arr) in zip(w, neigh):
        # Draw samples_per_neighbor per interpolated sample
        idxs = np.random.randint(0, arr.shape[0], size=(n_samples, samples_per_neighbor))
        draws = arr[idxs].mean(axis=1)   # average over samples_per_neighbor
        out += weight * draws

    return out


# -------------------------------------------------------
# 4. Posterior sampling using XGBoost model
# -------------------------------------------------------

def sample_posterior(model, xHI_true, logfX_true, feature_grid,
                     n_samples=2000, k=4, samples_per_neighbor=5):
    features = interpolate_features(
        xHI=xHI_true,
        logfX=logfX_true,
        grid=feature_grid,
        n_samples=n_samples,
        k=k,
        samples_per_neighbor=samples_per_neighbor
    )
    preds = model.predict(features)
    return preds  # shape (n_samples, 2)


# -------------------------------------------------------
# 5. Plot posterior
# -------------------------------------------------------

def plot_posterior(samples, true_point, outname):
    xHI = samples[:, 0]
    logfX = samples[:, 1]

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 4, 0.3], height_ratios=[4, 1, 0.3])

    # main 2D contour plot
    ax2d = fig.add_subplot(gs[0, 1])
    sns.kdeplot(x=logfX, y=xHI, fill=True, cmap="Blues", levels=10, thresh=0, ax=ax2d)
    ax2d.scatter([true_point[0]], [true_point[1]], color="red", s=80, label="True")
    ax2d.set_xlabel("log fX")
    ax2d.set_ylabel("xHI")
    ax2d.legend()

    # marginal: xHI
    ax_xHI = fig.add_subplot(gs[0, 2], sharey=ax2d)
    sns.kdeplot(y=xHI, fill=True, ax=ax_xHI, cmap="Blues")
    ax_xHI.get_xaxis().set_visible(False)

    # marginal: logfX
    ax_logfX = fig.add_subplot(gs[1, 1], sharex=ax2d)
    sns.kdeplot(x=logfX, fill=True, ax=ax_logfX, cmap="Blues")
    ax_logfX.get_yaxis().set_visible(False)

    fig.suptitle(f"Posterior for true point logfX={true_point[0]}, xHI={true_point[1]}")
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()
    logger.info(f"Saved {outname}")


# -------------------------------------------------------
# 6. MAIN DRIVER: generate posterior for 3 points
# -------------------------------------------------------

def run(model_path, features_dir, outdir):
    os.makedirs(outdir, exist_ok=True)

    model_raw, is_booster = load_model(model_path)
    model = ModelWrapper(model_raw, is_booster=is_booster)

    feature_grid = load_feature_grid(features_dir)

    points = [
        (-3.6, 0.80),
        (-3.6, 0.51),
        (-3.6, 0.24)
    ]

    for (logfX_true, xHI_true) in points:
        logger.info(f"Processing point (logfX={logfX_true}, xHI={xHI_true})")

        posterior = sample_posterior(
            model,
            xHI_true=xHI_true,
            logfX_true=logfX_true,
            feature_grid=feature_grid,
            n_samples=2000,
            k=4,
            samples_per_neighbor=5
        )

        outname = os.path.join(
            outdir,
            f"posterior_logfX_{logfX_true}_xHI_{xHI_true}.png"
        )

        plot_posterior(posterior,
                       true_point=(logfX_true, xHI_true),
                       outname=outname)


# -------------------------------------------------------
# Execute
# -------------------------------------------------------

if __name__ == "__main__":
    parser = base.setup_args_parser()

    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
    parser.add_argument('--latentdim', type=int, default=256, help='256, 512, etc')

    args = parser.parse_args()

    output_dir = base.create_output_dir(args=args)
    global logger
    logger = base.setup_logging(output_dir)


    run(
            model_path=args.modelfile,
        features_dir=args.datapath,
        outdir=output_dir
    )

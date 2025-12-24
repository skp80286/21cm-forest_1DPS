import os
import glob
import numpy as np
import csv
import re
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import f21_predict_base as base 

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters
        self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -4)

        self.bias_mu  = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -4)

        self.softplus = nn.Softplus()
        self.prior = torch.distributions.Normal(0, prior_sigma)

    def forward(self, x):
        weight_sigma = self.softplus(self.weight_rho)
        bias_sigma   = self.softplus(self.bias_rho)

        weight_eps = torch.randn_like(weight_sigma)
        bias_eps   = torch.randn_like(bias_sigma)

        W = self.weight_mu + weight_sigma * weight_eps
        b = self.bias_mu  + bias_sigma * bias_eps

        # KL divergence
        qw = torch.distributions.Normal(self.weight_mu, weight_sigma)
        qb = torch.distributions.Normal(self.bias_mu,  bias_sigma)

        kl = torch.distributions.kl.kl_divergence(qw, self.prior).sum()
        kl += torch.distributions.kl.kl_divergence(qb, self.prior).sum()

        return x @ W.T + b, kl



class BayesianNN(nn.Module):
    def __init__(self, hidden=[256, 128]):
        super().__init__()
        layers = []
        in_dim = 512

        for h in hidden:
            layers.append(BayesianLinear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(BayesianLinear(in_dim, 2))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl_total = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x, kl = layer(x)
                kl_total += kl
            else:
                x = layer(x)
        return x, kl_total

def load_data(datapath, quick=False, feature_group_size=1):
    """
    Loads all .npy files inside datapath.
    Each file contains 1000 samples of 512-dim latent features.
    The filename encodes logfX and xHI.
    Returns stacked (X, y) tensors.
    """
    X_list = []
    y_list = []
    files=sorted(glob.glob(datapath))
    if quick: files = files[:10]
    logger.info(f"Loading data from {len(files)} files.")
    for i, fname in enumerate(files):
        if not fname.endswith(".npy"):
            continue

        xHI, logfX = parse_labels_from_filename(fname)

        data = np.load(fname).astype(np.float32)
        # Expected shape: (1000, 512)
        if data.ndim != 2 or data.shape[1] != 512:
            raise ValueError(f"File {fname} has wrong shape: {data.shape}")

        if i == 0: logger.info(f"data.shape = {data.shape}")
        # Group features if specified
        if feature_group_size > 1:
            data = data.reshape(n // feature_group_size, feature_group_size, data.shape[1]).mean(axis=1)
            if i == 0: logger.info(f"After grouping, data.shape = {data.shape}")

        # Create labels for all 1000 samples
        labels = np.column_stack([
            np.full(data.shape[0], xHI, dtype=np.float32),
            np.full(data.shape[0], logfX, dtype=np.float32)
        ])

        X_list.append(data)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    logger.info(f"Loaded training data: X={X.shape}, y={y.shape}")

    return torch.tensor(X), torch.tensor(y)

def load_test_points(testdatapath, feature_group_size=1):
    """
    Loads test npy files with same naming convention.
    Returns a list of:
        - feature matrices (1000, 512)
        - (xHI, logfX) labels parsed from filename
    """
    features = []
    labels = []

    for fname in sorted(glob.glob(testdatapath)):
        if not fname.endswith(".npy"):
            continue

        xHI, logfX = parse_labels_from_filename(fname)

        data = np.load(fname).astype(np.float32)
        if data.ndim != 2 or data.shape[1] != 512:
            raise ValueError(f"Test file {fname} wrong shape: {data.shape}")

        features.append(torch.tensor(data))  # keep 1000 samples
        labels.append((xHI, logfX))

    logger.info(f"Loaded {len(features)} test points.")

    return features, labels

def create_loaders(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    N = len(dataset)

    val_size = int(0.2 * N)
    train_size = N - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, N

def train_model(model, train_loader, val_loader,
                lr=1e-3, epochs=40, beta=1.0, patience=6, N=1):
    
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            opt.zero_grad()

            pred, kl = model(xb)       # model returns (prediction, KL)
            nll = mse(pred, yb)

            loss = nll + beta * kl / N     # Variational Bayes loss

            loss.backward()
            opt.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred, kl = model(xb)
                nll = mse(pred, yb)
                val_loss += (nll + beta * kl / N).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model

def tune_hyperparams(X, y, quick=False):
    configs = [
        {"hidden": [256, 128], "lr": 1e-3},
        {"hidden": [512, 256], "lr": 5e-4},
        {"hidden": [512, 512], "lr": 1e-4},
    ]

    best_model = None
    best_config = None
    best_loss = 1e9

    for cfg in configs:
        logger.info(f"Testing config: {cfg}")

        model = BayesianNN(cfg["hidden"])
        train_loader, val_loader, N = create_loaders(X, y)

        epochs = 40
        if quick: epochs = 5
        trained = train_model(model, train_loader, val_loader, lr=cfg["lr"], epochs=epochs, N=N)

        # Compute validation loss
        mse = nn.MSELoss()
        val_loss = 0
        trained.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += mse(trained(xb)[0], yb).item()

        val_loss /= len(val_loader)
        logger.info(f"Val loss: {val_loss:.8f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = trained
            best_config = cfg

    return best_model, best_config

def save_model_config(output_dir, config, filename="config.json"):
    cfg_path = os.path.join(output_dir, filename)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {cfg_path}")

def load_model_config(output_dir, filename="config.json"):
    cfg_path = os.path.join(output_dir, filename)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg

def save_model(model, output_dir, filename="bnn_model.pt"):
    """
    Saves the trained BayesianNN model to output_dir/bnn_model.pt.
    Creates the directory if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)

    torch.save({
        "model_state_dict": model.state_dict()
    }, model_path)

    print(f"Model saved to: {model_path}")

def load_model(output_dir, config, filename="bnn_model.pt", device="cpu"):
    """
    Loads the BayesianNN model from output_dir/bnn_model.pt.
    hidden must match the architecture used during training.
    """
    model_path = os.path.join(output_dir, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    model = BayesianNN(config["hidden"])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Model loaded from: {model_path}")
    return model

def sample_posterior(model, x, num_samples=800):
    model.eval()
    preds = []
    for _ in range(num_samples):
        preds.append(model(x)[0].detach().cpu().numpy())
    return np.array(preds)  # shape: (S, 1, 2)

def save_posteriors_to_csv(filename, posteriors, true_params):
    """
    Save posterior samples for all test points to CSV.

    posteriors: list of arrays shaped (S,2)
                where S is number of posterior samples
                samples[:,0] = pred_xHI
                samples[:,1] = pred_logfX

    true_params: list of tuples (true_logfX, true_xHI)
                 supplied in same order as posteriors
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred_xHI", "pred_fX", "test_xHI", "test_fX"])

        for (samples, (true_xHI, true_logfX)) in zip(posteriors, true_params):
            
            for s in samples:
                pred_xHI = s[0]   # careful with ordering: depends on your code
                pred_fX  = s[1]

                # The user wants columns in terms of:
                # pred_xHI, pred_fX, test_xHI, test_fX

                writer.writerow([
                    pred_xHI,
                    pred_fX,
                    true_xHI,
                    true_logfX
                ])


def plot_all_posteriors(posteriors, true_params, posterior_means):
    """
    posteriors: list of arrays shaped (S, 2)  for each test point
    true_params: list of (logfX, xHI)
    posterior_means: list of (logfX_mean, xHI_mean)
    """

    logger.info(f'plot_all_posteriors: true_params={true_params}')
    colors = ["red", "green", "blue"]
    labels = [
        "Test Point: (0.80, -3.6)",
        "Test Point: (0.51, -3.6)",
        "Test Point: (0.24, -3.6)"
    ]

    plt.figure(figsize=(8, 6))

    for i, samples in enumerate(posteriors):
        color = colors[i]
        label = labels[i]

        # Extract posterior draws
        xHI_samples = samples[:, 0]
        logfX_samples = samples[:, 1]

        # KDE Contour plot
        sns.kdeplot(
            x=xHI_samples,
            y=logfX_samples,
            levels=10,
            color=color,
            linewidths=1.5,
            alpha=0.8,
            label=label if i == 0 else None
        )

        # True value (star)
        true_xHI, true_logfX = true_params[i]
        plt.scatter(
            true_xHI, true_logfX,
            color=color,
            s=120,
            marker="*",
            edgecolor="black",
            linewidth=1.2,
            label="True value" if i == 0 else None
        )

        # Posterior mean (circle)
        mean_xHI, mean_logfX = posterior_means[i]
        plt.scatter(
            mean_xHI, mean_logfX,
            color=color,
            s=80,
            marker="o",
            edgecolor="black",
            linewidth=1.2,
            label="Posterior mean" if i == 0 else None
        )

    plt.xlabel("xHI")
    plt.ylabel("logfX")
    plt.title("Posterior Distributions for Three Test Points")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'posterior_all.png'), format='png', dpi=300)


def plot_posterior(samples, title, true_params, output_dir):
    xHI = samples[:,0]
    logfX = samples[:,1]

    sns.kdeplot(x=xHI, y=logfX, fill=True, cmap="mako", levels=30)
    plt.xlabel("xHI")
    plt.ylabel("logfX")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f'posterior_{true_params[0]:.2f}_{true_params[1]:.2f}.png'), format='png', dpi=300)

def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def run(datapath, testdatapath, quick=False, output_dir="tmp_out", run_mode="train_test", model_dir=None):
    # Load training data
    X, y = load_data(datapath, quick=quick)

    if run_mode == "train_test":
        # Tune and train
        model, config = tune_hyperparams(X, y, quick=quick)
        print(f'Best config is: {config}\nBest model is: {model}')

        save_model(model, output_dir)
        save_model_config(output_dir, config)
        model_dir = output_dir

    config = load_model_config(model_dir)
    model = load_model(model_dir, config=config)

    # Metrics on full train set
    model.eval()
    with torch.no_grad():
        pred_train = model(X)[0]
    r2, rmse = compute_metrics(y, pred_train)
    logger.info(f"Train R2: {r2:.8f}")
    logger.info(f"Train RMSE: {rmse:.8f}")

    # Load test points
    X_test, y_test = load_test_points(testdatapath)
    logger.info(f"Loaded test points: {y_test}, Total:{len(X_test)}")
    #logger.info(f"Test data: {X_test}")
    
    posteriors = []        # list of (800,2)
    posterior_means = []   # list of (2,)

    for i in range(len(X_test)):
        x = X_test[i] 
        y = y_test[i] 
        #logger.info(f'type(x)={type(x)},\nx={x}')
        samples = sample_posterior(model, x, num_samples=800)
        #samples = sample_posterior(model, feat) 
        samples = samples[:,0,:]   # shape (S, 2)

        plot_posterior(samples, f"Posterior for test point {y}", y, output_dir)

        logger.info(f"Posterior mean for test point {i+1} {y}:")
        #logger.info(f"xHI = {samples[:,0]}")
        #logger.info(f"logfX = {samples[:,1]}")
        logger.info(f"xHI = {samples[:,0].mean():.8f}")
        logger.info(f"logfX = {samples[:,1].mean():.8f}")
        posteriors.append(samples)
        posterior_means.append( (samples[:,0].mean(), samples[:,1].mean()) )

    save_posteriors_to_csv(f'{output_dir}/test_results.csv', posteriors, y_test)
    plot_all_posteriors(posteriors, y_test, posterior_means)

def parse_labels_from_filename(filename):
    """
    Extract logfX and xHI from filenames using regex.
    Works for patterns like:
    - ...xHI0.51_logfX-3.6.npy
    - ...fX-3.6_xHI0.51.npy
    """
    xHI_match   = re.search(r"xHI(\d+\.?\d*)", filename)
    logfX_match = re.search(r"fX(-?\d+\.?\d*)", filename)

    if logfX_match is None or xHI_match is None:
        raise ValueError(f"Cannot parse labels from filename: {filename}")

    xHI   = float(xHI_match.group(1))
    logfX = float(logfX_match.group(1))

    return xHI, logfX

# -------------------------------------------------------
# Execute
# -------------------------------------------------------

if __name__ == "__main__":
    parser = base.setup_args_parser()

    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
    parser.add_argument('--latentdim', type=int, default=256, help='256, 512, etc')
    parser.add_argument('--feature_group_size', type=int, default=1, help='1, 10, etc. how many samples to group ')
    parser.add_argument('--quick', action='store_true', help='quick run for checking code')

    args = parser.parse_args()

    output_dir = base.create_output_dir(args=args)
    global logger
    logger = base.setup_logging(output_dir)


    run(datapath=f"{args.datapath}/*PSO*rms3.3*.npy",
        testdatapath=f"{args.testdatapath}/*fX-3.6*PSO*rms3.3*.npy", quick=args.quick, output_dir=output_dir)

    logger.info(f"Completed run. output: {output_dir}")


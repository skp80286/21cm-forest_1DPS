import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging


# ----------------------------------------------------------------------
# Utility: activation & pooling factories
# ----------------------------------------------------------------------
def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_pooling(pool_type: str):
    pool_type = pool_type.lower()
    if pool_type == "max":
        return nn.MaxPool1d(kernel_size=2)
    elif pool_type == "avg":
        return nn.AvgPool1d(kernel_size=2)
    else:
        raise ValueError(f"Unknown pooling type: {pool_type}")


# ----------------------------------------------------------------------
# Bayesian-ish CNN using MC Dropout
# ----------------------------------------------------------------------
class BayesianCNN21cm(nn.Module):
    """
    1D CNN for 21-cm spectrum -> (xHI, logfX)
    Uses dropout at train & test time (MC Dropout) to approximate Bayesian NN.
    Output is mapped into:
      xHI   ∈ [0, 1]
      logfX ∈ [-4, 1]
    """

    def __init__(
        self,
        input_length=3584,
        in_channels=1,
        conv_channels=(32, 64, 128, 256),
        kernel_sizes=((5, 3), (5, 3), (5, 3), (5, 3)),
        fc_layers=(128, 64),
        activation="relu",
        pooling="max",
        dropout=0.1,
    ):
        super().__init__()

        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have same length."

        self.input_length = input_length
        self.activation_name = activation
        act = get_activation(activation)

        conv_blocks = []
        prev_c = in_channels

        for out_c, ks in zip(conv_channels, kernel_sizes):
            k1, k2 = ks
            block = nn.Sequential(
                nn.Conv1d(prev_c, out_c, kernel_size=k1, padding=k1 // 2),
                act,
                nn.Conv1d(out_c, out_c, kernel_size=k2, padding=k2 // 2),
                act,
                get_pooling(pooling),
                nn.Dropout(dropout),  # MC dropout in conv block
            )
            conv_blocks.append(block)
            prev_c = out_c

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Fully connected part
        fc = []
        in_features = conv_channels[-1]
        for hidden in fc_layers:
            fc.append(nn.Linear(in_features, hidden))
            fc.append(get_activation(activation))
            fc.append(nn.Dropout(dropout))  # MC dropout in FC part
            in_features = hidden

        # Final layer: 2 raw outputs -> transform to xHI, logfX ranges
        fc.append(nn.Linear(in_features, 2))
        self.regressor = nn.Sequential(*fc)

        # Priors / ranges for parameters
        self.xHI_min, self.xHI_max = 0.0, 1.0
        self.logfX_min, self.logfX_max = -4.0, 1.0

    def forward(self, x):
        """
        x: (batch, 1, L) or (batch, L)
        returns: (batch, 2) with columns [xHI, logfX]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,L)

        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        raw = self.regressor(x)  # (B,2)

        # Map outputs into desired ranges
        #   xHI   = sigmoid(raw0) * (1 - 0) + 0
        #   logfX = sigmoid(raw1) * (1 - (-4)) + (-4) = sigmoid(raw1)*5 - 4
        xHI = torch.sigmoid(raw[:, 0])
        logfX = self.logfX_min + (self.logfX_max - self.logfX_min) * torch.sigmoid(
            raw[:, 1]
        )

        out = torch.stack([xHI, logfX], dim=-1)
        return out


# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------
def create_bayesian_model(
    input_length=3584,
    in_channels=1,
    conv_channels=(32, 64, 128, 256),
    kernel_sizes=((5, 3), (5, 3), (5, 3), (5, 3)),
    fc_layers=(128, 64),
    activation="relu",
    pooling="max",
    dropout=0.1,
):
    return BayesianCNN21cm(
        input_length=input_length,
        in_channels=in_channels,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        fc_layers=fc_layers,
        activation=activation,
        pooling=pooling,
        dropout=dropout,
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_model(
    model,
    train_dataset,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1:03d}/{num_epochs:03d} - train_loss: {epoch_loss:.6e}")

    return model


# ----------------------------------------------------------------------
# Point-estimate testing (just for quick MSE/MAE)
# ----------------------------------------------------------------------
def test_model(
    model,
    test_dataset,
    batch_size=256,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds, trues = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = model(inputs)
            preds.append(outputs.cpu())
            trues.append(targets.cpu())

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    return y_true, y_pred


# ----------------------------------------------------------------------
# Simple regression metrics
# ----------------------------------------------------------------------
def report_test_scores(y_true: torch.Tensor, y_pred: torch.Tensor):
    diff = y_pred - y_true
    mse_overall = torch.mean(diff ** 2).item()
    mae_overall = torch.mean(torch.abs(diff)).item()

    mse_per_param = torch.mean(diff ** 2, dim=0)
    mae_per_param = torch.mean(torch.abs(diff), dim=0)

    logging.info("=== Test Scores ===")
    logging.info(f"Overall MSE: {mse_overall:.6e}")
    logging.info(f"Overall MAE: {mae_overall:.6e}")

    for i, (mse_i, mae_i) in enumerate(zip(mse_per_param, mae_per_param)):
        logging.info(f"Param {i}: MSE = {mse_i.item():.6e}, MAE = {mae_i.item():.6e}")

    return {
        "mse_overall": mse_overall,
        "mae_overall": mae_overall,
        "mse_per_param": mse_per_param.numpy(),
        "mae_per_param": mae_per_param.numpy(),
    }


# ----------------------------------------------------------------------
# Posterior sampling via MC Dropout
# ----------------------------------------------------------------------
def sample_posterior(
    model: nn.Module,
    spectrum: torch.Tensor,
    n_samples: int = 500,
    device=None,
):
    """
    spectrum: shape (L,) or (1, L) or (1, 1, L)
    returns: samples of shape (n_samples, 2) for [xHI, logfX]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.train()  # IMPORTANT: keep dropout ON

    # Prepare input
    spec = spectrum
    if isinstance(spec, (list, tuple)):
        spec = torch.tensor(spec, dtype=torch.float32)
    spec = spec.to(device, dtype=torch.float32)

    if spec.ndim == 1:
        spec = spec.unsqueeze(0)        # (1, L)
    if spec.ndim == 2:
        spec = spec.unsqueeze(1)        # (1, 1, L)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(spec)           # (1,2)
            samples.append(out.squeeze(0).cpu())  # (2,)

    samples = torch.stack(samples, dim=0)  # (n_samples, 2)
    return samples


# ----------------------------------------------------------------------
# Posterior plotting
# ----------------------------------------------------------------------
def plot_posterior(
    samples: torch.Tensor,
    true_values=None,  # e.g., (xHI_true, logfX_true)
    param_names=("xHI", "logfX"),
    bins=50,
    output_dir="./tmp_out"
):
    """
    samples: (n_samples, 2) tensor for [xHI, logfX]
    """
    if isinstance(samples, torch.Tensor):
        samples_np = samples.numpy()
    else:
        samples_np = samples

    x = samples_np[:, 0]
    y = samples_np[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D hist for xHI
    ax = axes[0]
    ax.hist(x, bins=bins, density=True, alpha=0.7)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel("Posterior density")
    if true_values is not None:
        ax.axvline(true_values[0], color="k", linestyle="--", label="True")
        ax.legend()

    # 1D hist for logfX
    ax = axes[1]
    ax.hist(y, bins=bins, density=True, alpha=0.7)
    ax.set_xlabel(param_names[1])
    ax.set_ylabel("Posterior density")
    if true_values is not None:
        ax.axvline(true_values[1], color="k", linestyle="--", label="True")
        ax.legend()

    # 2D joint scatter
    ax = axes[2]
    ax.scatter(x, y, s=5, alpha=0.3)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    if true_values is not None:
        ax.scatter([true_values[0]], [true_values[1]],
                   marker="x", color="red", s=100, label="True")
        ax.legend()

    plt.tight_layout()
    imagepath = os.path.join(output_dir, 'posterior.png')
    plt.savefig(imagepath, format='png', dpi=300)



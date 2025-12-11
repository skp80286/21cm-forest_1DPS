import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
import json


# ----------------------------------------------------------------------
# Utility: activation function factory
# ----------------------------------------------------------------------
def get_activation(name: str):
    """Return a PyTorch activation module based on string name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    else:
        raise ValueError(f"Unknown activation function: {name}")


# Utility: pooling factory
def get_pooling(pool_type: str):
    """Return a pooling module based on string name."""
    pool_type = pool_type.lower()
    if pool_type == "max":
        return nn.MaxPool1d(kernel_size=2)
    elif pool_type == "avg":
        return nn.AvgPool1d(kernel_size=2)
    else:
        raise ValueError(f"Unknown pooling type: {pool_type}")


# ----------------------------------------------------------------------
# CNN Model
# ----------------------------------------------------------------------
class CNN21cm(nn.Module):
    def __init__(
        self,
        input_length=3584,
        in_channels=1,
        conv_channels=(32, 64, 128, 256),
        kernel_sizes=((5, 3), (5, 3), (5, 3), (5, 3)),
        fc_layers=(128, 64),
        output_dim=2,
        activation="relu",
        pooling="max",
        dropout=0.1,
    ):
        super().__init__()

        # -------------------------------------------------------
        # Save ALL init arguments as attributes so getattr works
        # -------------------------------------------------------
        self.input_length = input_length
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.fc_layers = fc_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.pooling = pooling
        self.dropout = dropout


        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have same length."

        act = get_activation(activation)
        pool = get_pooling(pooling)

        conv_blocks = []
        prev_c = in_channels

        for out_c, ks in zip(conv_channels, kernel_sizes):
            k1, k2 = ks
            block = nn.Sequential(
                nn.Conv1d(prev_c, out_c, kernel_size=k1, padding=k1 // 2),
                act,
                nn.Conv1d(out_c, out_c, kernel_size=k2, padding=k2 // 2),
                act,
                pool,  # configurable pooling
            )
            conv_blocks.append(block)
            prev_c = out_c

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # global pooling
        self.flatten = nn.Flatten()

        # Fully connected layers
        fc = []
        in_features = conv_channels[-1]
        for hidden in fc_layers:
            fc.append(nn.Linear(in_features, hidden))
            fc.append(get_activation(activation))
            fc.append(nn.Dropout(dropout))
            in_features = hidden
        fc.append(nn.Linear(in_features, output_dim))
        self.regressor = nn.Sequential(*fc)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,L)
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x


# ----------------------------------------------------------------------
# 1. Model Factory
# ----------------------------------------------------------------------
def create_model(
    input_length=3584,
    in_channels=1,
    conv_channels=(32, 64, 128, 256),
    kernel_sizes=((5, 3), (5, 3), (5, 3), (5, 3)),
    fc_layers=(128, 64),
    output_dim=2,
    activation="relu",
    pooling="max",
    dropout=0.1,
):
    return CNN21cm(
        input_length=input_length,
        in_channels=in_channels,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        fc_layers=fc_layers,
        output_dim=output_dim,
        activation=activation,
        pooling=pooling,
        dropout=dropout,
    )


# ----------------------------------------------------------------------
# 2. Training
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
        logging.info(f"Epoch {epoch+1:03d}/{num_epochs:03d} - Train Loss: {epoch_loss:.6f}")

    return model


# ----------------------------------------------------------------------
# 3. Testing
# ----------------------------------------------------------------------
def test_model(model, test_dataset, batch_size=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds, trues = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = model(inputs)
            preds.append(outputs.cpu())
            trues.append(targets.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)

    return y_true, y_pred


# ----------------------------------------------------------------------
# 4. Reporting Metrics
# ----------------------------------------------------------------------
def report_test_scores(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute and print regression metrics on test predictions.
    Returns MSE, MAE, and R2 (overall and per-parameter).
    """

    diff = y_pred - y_true
    mse_overall = torch.mean(diff ** 2).item()
    mae_overall = torch.mean(torch.abs(diff)).item()

    mse_per_param = torch.mean(diff ** 2, dim=0)
    mae_per_param = torch.mean(torch.abs(diff), dim=0)

    # ------------------------
    # R² Score Calculation
    # ------------------------
    var_total = torch.var(y_true, dim=0, unbiased=False)  # variance of true values
    r2_per_param = 1.0 - mse_per_param / var_total

    # Weighted overall R² across both outputs
    r2_overall = 1.0 - torch.mean(mse_per_param / var_total).item()

    print("=== Test Scores ===")
    print(f"Overall MSE: {mse_overall:.6e}")
    print(f"Overall MAE: {mae_overall:.6e}")
    print(f"Overall R² : {r2_overall:.6f}")
    print()

    for i in range(len(mse_per_param)):
        print(f"Param {i}:")
        print(f"  MSE = {mse_per_param[i].item():.6e}")
        print(f"  MAE = {mae_per_param[i].item():.6e}")
        print(f"  R²  = {r2_per_param[i].item():.6f}")
        print()

    return {
        "mse_overall": mse_overall,
        "mae_overall": mae_overall,
        "r2_overall": r2_overall,
        "mse_per_param": mse_per_param.numpy(),
        "mae_per_param": mae_per_param.numpy(),
        "r2_per_param": r2_per_param.numpy(),
    }


def save_model(model, path, config=None, metadata=None):
    """
    Save CNN21cm or BayesianCNN21cm model.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    path : str
        Base path without extension. Saves:
          path + ".pt"      → model weights
          path + "_config.json" → model config
          path + "_meta.json"   → metadata (optional)
    config : dict (optional)
        Model initialization arguments.
        If None, attempts to infer from model.__dict__.
    metadata : dict (optional)
        Extra info such as training loss, epoch count, etc.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), path + ".pt")
    print(f"[Model] Weights saved to {path}.pt")

    # Save config (if provided or inferable)
    if config is None:
        # Automatically infer init parameters from the model object
        config = {
            "input_length": getattr(model, "input_length", None),
            "in_channels": 1,
            "conv_channels": getattr(model, "conv_channels", None)
            if hasattr(model, "conv_channels") else None,
            "kernel_sizes": getattr(model, "kernel_sizes", None),
            "fc_layers": getattr(model, "fc_layers", None),
            "activation": getattr(model, "activation_name", None),
            "pooling": getattr(model, "pooling", None)
            if hasattr(model, "pooling") else None,
            "dropout": getattr(model, "dropout", None)
            if hasattr(model, "dropout") else None,
        }

    with open(path + "_config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"[Model] Config saved to {path}_config.json")

    # Save metadata if provided
    if metadata is not None:
        with open(path + "_meta.json", "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[Model] Metadata saved to {path}_meta.json")


def load_model(path, device=None, load_metadata=True):
    """
    Load CNN21cm or BayesianCNN21cm model from disk.

    Parameters
    ----------
    path : str
        Base path without extension (same as used in save_model).
        Loads:
            path.pt
            path_config.json
            path_meta.json   (optional)
    device : str or torch.device
        "cpu", "cuda", etc. Defaults to GPU if available.
    load_metadata : bool
        If True, loads metadata from path_meta.json (if exists).

    Returns
    -------
    model : nn.Module
        Reconstructed model with weights loaded.
    metadata : dict or None
        Training metadata if available.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # --------------------------------------------------------------
    # 1. Load configuration JSON
    # --------------------------------------------------------------
    config_path = path + "_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Convert lists back to tuples if needed
    if isinstance(config.get("conv_channels"), list):
        config["conv_channels"] = tuple(config["conv_channels"])
    if isinstance(config.get("fc_layers"), list):
        config["fc_layers"] = tuple(config["fc_layers"])
    if isinstance(config.get("kernel_sizes"), list):
        config["kernel_sizes"] = [tuple(k) for k in config["kernel_sizes"]]

    # --------------------------------------------------------------
    # 2. Create the correct model class
    # --------------------------------------------------------------
    model = CNN21cm(
        input_length=config["input_length"],
        in_channels=config["in_channels"],
        conv_channels=config["conv_channels"],
        kernel_sizes=config["kernel_sizes"],
        fc_layers=config["fc_layers"],
        activation=config["activation"],
        pooling=config["pooling"],
        dropout=config["dropout"],
    )

    # --------------------------------------------------------------
    # 3. Load weights
    # --------------------------------------------------------------
    weights_path = path + ".pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"[Model] Loaded model from {weights_path}")

    # --------------------------------------------------------------
    # 4. Load metadata (optional)
    # --------------------------------------------------------------
    metadata = None
    if load_metadata:
        meta_path = path + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            print(f"[Model] Loaded metadata from {meta_path}")

    return model, metadata


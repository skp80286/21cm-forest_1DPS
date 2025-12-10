import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging


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
def report_test_scores(y_true, y_pred):
    diff = y_pred - y_true
    mse_overall = torch.mean(diff ** 2).item()
    mae_overall = torch.mean(torch.abs(diff)).item()

    mse_per_param = torch.mean(diff ** 2, dim=0)
    mae_per_param = torch.mean(torch.abs(diff), dim=0)

    logging.info("=== Test Scores ===")
    logging.info(f"Overall MSE: {mse_overall:.6e}")
    logging.info(f"Overall MAE: {mae_overall:.6e}")

    for i in range(len(mse_per_param)):
        logging.info(f"Param {i}: MSE={mse_per_param[i]:.6e}, MAE={mae_per_param[i]:.6e}")

    return {
        "mse_overall": mse_overall,
        "mae_overall": mae_overall,
        "mse_per_param": mse_per_param.numpy(),
        "mae_per_param": mae_per_param.numpy(),
    }



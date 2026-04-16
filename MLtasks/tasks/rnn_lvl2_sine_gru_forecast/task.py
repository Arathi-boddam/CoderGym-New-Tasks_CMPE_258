import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "rnn_lvl2_sine_gru_forecast"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_task_metadata():
    return {
        "task_id": TASK_ID,
        "task_type": "sequence_regression",
        "dataset": "synthetic_noisy_sine_windows",
        "features": ["gru", "onecyclelr", "smooth_l1", "gradient_clipping"],
        "input_length": 30,
        "output_dim": 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_windows(n_samples, seq_len, seed):
    rng = np.random.RandomState(seed)
    x = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    base_t = np.linspace(0.0, 1.0, seq_len + 1, dtype=np.float32)
    for i in range(n_samples):
        freq = rng.uniform(0.8, 3.2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.6, 1.4)
        trend = rng.uniform(-0.2, 0.2) * base_t
        wave = amp * np.sin(2.0 * np.pi * freq * base_t + phase) + trend
        noise = rng.normal(0.0, 0.03, size=seq_len + 1)
        series = (wave + noise).astype(np.float32)
        x[i, :, 0] = series[:-1]
        y[i, 0] = series[-1]
    return x, y


def make_dataloaders(batch_size=64):
    seq_len = 30
    x, y = _make_windows(3200, seq_len, seed=42)
    split = int(0.8 * len(x))
    train_ds = TensorDataset(torch.tensor(x[:split]), torch.tensor(y[:split]))
    val_ds = TensorDataset(torch.tensor(x[split:]), torch.tensor(y[split:]))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        seq_len,
        1,
    )


class SineGRU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.05)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def build_model(seq_len, output_dim, device):
    return SineGRU().to(device)


def train(model, train_loader, val_loader, device, epochs=30):
    criterion = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    history = {"train_loss": [], "val_mse": [], "val_r2": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for seq, target in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_mse"].append(val_metrics["mse"])
        history["val_r2"].append(val_metrics["r2"])
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} val_mse={val_metrics['mse']:.5f} val_r2={val_metrics['r2']:.4f}")
    return history


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for seq, target in data_loader:
            preds.append(model(seq.to(device)).cpu().numpy())
            targets.append(target.numpy())
    preds_arr = np.concatenate(preds).reshape(-1)
    targets_arr = np.concatenate(targets).reshape(-1)
    return {
        "mse": float(mean_squared_error(targets_arr, preds_arr)),
        "mae": float(mean_absolute_error(targets_arr, preds_arr)),
        "r2": float(r2_score(targets_arr, preds_arr)),
    }


def predict(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for seq, _ in data_loader:
            outputs.append(model(seq.to(device)).cpu().numpy())
    return np.concatenate(outputs).reshape(-1)


def save_artifacts(model, history, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    set_seed(42)
    device = get_device()
    train_loader, val_loader, seq_len, output_dim = make_dataloaders()
    model = build_model(seq_len, output_dim, device)
    history = train(model, train_loader, val_loader, device)
    metrics = {"train": evaluate(model, train_loader, device), "val": evaluate(model, val_loader, device)}
    save_artifacts(model, history, metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["val"]["r2"] > 0.85 and metrics["val"]["mse"] < 0.04 else 1


if __name__ == "__main__":
    sys.exit(main())


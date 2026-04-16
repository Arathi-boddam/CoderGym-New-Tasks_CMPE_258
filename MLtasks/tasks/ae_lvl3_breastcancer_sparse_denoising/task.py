import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "ae_lvl3_breastcancer_sparse_denoising"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_task_metadata():
    return {
        "task_id": TASK_ID,
        "task_type": "autoencoder",
        "dataset": "sklearn_breast_cancer_features",
        "features": ["denoising", "sparse_latent_l1", "reducelronplateau"],
        "input_dim": 30,
        "latent_dim": 10,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=32):
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    x_train, x_val = train_test_split(x, test_size=0.2, random_state=42, stratify=data.target)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(x_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(x_val))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        x_train.shape[1],
        10,
    )


class SparseDenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.SiLU(),
            nn.Linear(48, latent_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.SiLU(),
            nn.Linear(48, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def build_model(input_dim, latent_dim, device):
    return SparseDenoisingAE(input_dim, latent_dim).to(device)


def _corrupt(x, noise_std=0.08, drop_prob=0.08):
    noisy = x + noise_std * torch.randn_like(x)
    keep = (torch.rand_like(x) > drop_prob).float()
    return noisy * keep


def train(model, train_loader, val_loader, device, epochs=120):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    history = {"train_loss": [], "val_mse": [], "val_r2": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for clean, target in train_loader:
            clean = clean.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, latent = model(_corrupt(clean))
            loss = torch.mean((recon - target) ** 2) + 1e-4 * torch.mean(torch.abs(latent))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mse"])
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_mse"].append(val_metrics["mse"])
        history["val_r2"].append(val_metrics["r2"])
        if (epoch + 1) % 30 == 0:
            print(f"epoch={epoch + 1} val_mse={val_metrics['mse']:.4f} val_r2={val_metrics['r2']:.4f}")
    return history


def evaluate(model, data_loader, device):
    model.eval()
    recons, targets = [], []
    with torch.no_grad():
        for clean, target in data_loader:
            recon, _ = model(clean.to(device))
            recons.append(recon.cpu().numpy())
            targets.append(target.numpy())
    recon_arr = np.concatenate(recons, axis=0).reshape(-1)
    target_arr = np.concatenate(targets, axis=0).reshape(-1)
    return {
        "mse": float(mean_squared_error(target_arr, recon_arr)),
        "mae": float(mean_absolute_error(target_arr, recon_arr)),
        "r2": float(r2_score(target_arr, recon_arr)),
    }


def predict(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for clean, _ in data_loader:
            recon, _ = model(clean.to(device))
            outputs.append(recon.cpu().numpy())
    return np.concatenate(outputs, axis=0)


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
    train_loader, val_loader, input_dim, latent_dim = make_dataloaders()
    model = build_model(input_dim, latent_dim, device)
    history = train(model, train_loader, val_loader, device)
    metrics = {"train": evaluate(model, train_loader, device), "val": evaluate(model, val_loader, device)}
    save_artifacts(model, history, metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["val"]["r2"] > 0.78 and metrics["val"]["mse"] < 0.22 else 1


if __name__ == "__main__":
    sys.exit(main())


import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "mlp_lvl2_moons_mixup"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_task_metadata():
    return {
        "task_id": TASK_ID,
        "task_type": "classification",
        "dataset": "sklearn_make_moons",
        "features": ["mixup", "adamw", "cosine_lr", "gradient_clipping"],
        "input_dim": 2,
        "num_classes": 2,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=64):
    x, y = make_moons(n_samples=2500, noise=0.22, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x.astype(np.float32),
        y.astype(np.int64),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val, dtype=torch.long))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        x_train.shape[1],
        2,
    )


class MoonMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim, num_classes, device):
    return MoonMLP(input_dim, num_classes).to(device)


def _mixup(features, labels, num_classes, alpha=0.25):
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(features.size(0), device=features.device)
    mixed = lam * features + (1.0 - lam) * features[index]
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_onehot[index]
    return mixed, mixed_labels


def train(model, train_loader, val_loader, device, epochs=80):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "val_accuracy": [], "val_f1_macro": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mixed_x, mixed_y = _mixup(features, labels, 2)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mixed_x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(mixed_y * log_probs).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        if (epoch + 1) % 20 == 0:
            print(
                f"epoch={epoch + 1} train_loss={history['train_loss'][-1]:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
            )

    return history


def evaluate(model, data_loader, device):
    model.eval()
    probs_all, preds_all, targets_all = [], [], []
    with torch.no_grad():
        for features, labels in data_loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            preds_all.append(np.argmax(probs, axis=1))
            targets_all.append(labels.numpy())

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    onehot = np.eye(probs.shape[1])[targets]
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro")),
        "mse": float(mean_squared_error(onehot.reshape(-1), probs.reshape(-1))),
        "r2": float(r2_score(onehot.reshape(-1), probs.reshape(-1))),
    }


def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for features, _ in data_loader:
            preds.append(torch.argmax(model(features.to(device)), dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


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
    train_loader, val_loader, input_dim, num_classes = make_dataloaders()
    model = build_model(input_dim, num_classes, device)
    history = train(model, train_loader, val_loader, device)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    metrics = {"train": train_metrics, "val": val_metrics}
    save_artifacts(model, history, metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if val_metrics["accuracy"] > 0.94 and val_metrics["f1_macro"] > 0.94 else 1


if __name__ == "__main__":
    sys.exit(main())


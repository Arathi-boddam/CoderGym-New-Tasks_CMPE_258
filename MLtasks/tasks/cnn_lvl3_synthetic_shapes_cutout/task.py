import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "cnn_lvl3_synthetic_shapes_cutout"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_task_metadata():
    return {
        "task_id": TASK_ID,
        "task_type": "classification",
        "dataset": "synthetic_16x16_shapes",
        "features": ["cutout", "label_smoothing", "adamw", "cosine_lr"],
        "image_shape": [1, 16, 16],
        "num_classes": 4,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_image(label, rng):
    img = rng.normal(0.0, 0.05, size=(16, 16)).astype(np.float32)
    shift = rng.randint(-2, 3)
    if label == 0:
        row = np.clip(8 + shift, 2, 13)
        img[row - 1 : row + 2, 2:14] += 1.0
    elif label == 1:
        col = np.clip(8 + shift, 2, 13)
        img[2:14, col - 1 : col + 2] += 1.0
    elif label == 2:
        for i in range(3, 13):
            j = np.clip(i + shift, 1, 14)
            img[i - 1 : i + 2, j - 1 : j + 2] += 0.8
    else:
        top = np.clip(5 + shift, 2, 9)
        img[top : top + 6, top : top + 6] += 0.85
    return np.clip(img, 0.0, 1.0)


def make_dataloaders(batch_size=64):
    rng = np.random.RandomState(42)
    labels = np.repeat(np.arange(4), 700)
    images = np.stack([_make_image(int(label), rng) for label in labels], axis=0)[:, None, :, :]
    x_train, x_val, y_train, y_val = train_test_split(
        images.astype(np.float32),
        labels.astype(np.int64),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val, dtype=torch.long))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        (1, 16, 16),
        4,
    )


class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(image_shape, num_classes, device):
    return ShapeCNN(num_classes).to(device)


def _cutout(x, size=4):
    x = x.clone()
    b, _, h, w = x.shape
    for i in range(b):
        cy = torch.randint(0, h, (1,), device=x.device).item()
        cx = torch.randint(0, w, (1,), device=x.device).item()
        y0, y1 = max(0, cy - size // 2), min(h, cy + size // 2)
        x0, x1 = max(0, cx - size // 2), min(w, cx + size // 2)
        x[i, :, y0:y1, x0:x1] = 0.0
    return x


def train(model, train_loader, val_loader, device, epochs=35):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "val_accuracy": [], "val_f1_macro": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = _cutout(images.to(device))
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}")
    return history


def evaluate(model, data_loader, device):
    model.eval()
    probs_all, preds_all, targets_all = [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
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
        for images, _ in data_loader:
            preds.append(torch.argmax(model(images.to(device)), dim=1).cpu().numpy())
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
    train_loader, val_loader, image_shape, num_classes = make_dataloaders()
    model = build_model(image_shape, num_classes, device)
    history = train(model, train_loader, val_loader, device)
    metrics = {"train": evaluate(model, train_loader, device), "val": evaluate(model, val_loader, device)}
    save_artifacts(model, history, metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["val"]["accuracy"] > 0.97 and metrics["val"]["f1_macro"] > 0.97 else 1


if __name__ == "__main__":
    sys.exit(main())


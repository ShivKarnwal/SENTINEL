
import csv
import json
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/raw/PlantVillage"
SPLIT_DIR = ROOT / "datasets/splits/grouped"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

torch.manual_seed(42)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")


class PlantDataset(Dataset):
    def __init__(self, split_file, class_to_idx, transform):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        with open(split_file, "r") as file:
            for line in file:
                relative_path = line.strip()
                if not relative_path:
                    continue

                parts = relative_path.split("/")
                class_name = parts[2]

                self.samples.append(
                    (DATA_ROOT / relative_path, class_to_idx[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    split_files = {
        name: SPLIT_DIR / f"{name}.txt"
        for name in ["train", "validation", "test"]
    }

    for path in split_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    train_paths = [
        line.strip()
        for line in split_files["train"].read_text().splitlines()
        if line.strip()
    ]

    class_names = sorted({
        path.split("/")[2]
        for path in train_paths
    })

    class_to_idx = {
        name: index for index, name in enumerate(class_names)
    }

    with open(REPORT_DIR / "class_mapping.json", "w") as file:
        json.dump(class_to_idx, file, indent=4)

    weights = MobileNet_V3_Small_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])

    eval_transform = weights.transforms()

    train_dataset = PlantDataset(
        split_files["train"], class_to_idx, train_transform
    )
    val_dataset = PlantDataset(
        split_files["validation"], class_to_idx, eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Device: {DEVICE}")
    print(f"Classes: {len(class_names)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("Loading pretrained MobileNetV3-Small...")

    model = models.mobilenet_v3_small(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, len(class_names))

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0
    history = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        elapsed = time.time() - start_time

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "seconds": elapsed
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "architecture": "MobileNetV3-Small",
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch + 1,
                "validation_accuracy": val_accuracy,
            }, MODEL_DIR / "sentinel_mobilenetv3_small_best.pth")

            print("Best model checkpoint saved.", flush=True)

        with open(REPORT_DIR / "training_history.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print("\nTRAINING COMPLETE")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("Best checkpoint: models/sentinel_mobilenetv3_small_best.pth")


if __name__ == "__main__":
    main()

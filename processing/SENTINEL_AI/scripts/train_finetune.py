
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/raw/PlantVillage"
SPLIT_DIR = ROOT / "datasets/splits/grouped"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

BASELINE_PATH = MODEL_DIR / "sentinel_mobilenetv3_small_best.pth"
OUTPUT_PATH = MODEL_DIR / "sentinel_mobilenetv3_small_finetuned_best.pth"

BATCH_SIZE = 8
EPOCHS = 2
FEATURE_LR = 0.00001
CLASSIFIER_LR = 0.0001
NUM_WORKERS = 0
UNFREEZE_FROM = 9

torch.manual_seed(42)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")


class PlantDataset(Dataset):
    def __init__(self, split_file, class_to_idx, transform):
        self.samples = []
        self.transform = transform

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

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {BASELINE_PATH}"
        )

    train_file = SPLIT_DIR / "train.txt"
    val_file = SPLIT_DIR / "validation.txt"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("Training or validation split is missing.")

    print("Loading original baseline checkpoint...", flush=True)

    checkpoint = torch.load(
        BASELINE_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    train_paths = [
        line.strip()
        for line in train_file.read_text().splitlines()
        if line.strip()
    ]

    actual_classes = {
        path.split("/")[2] for path in train_paths
    }

    if actual_classes != set(class_to_idx):
        raise ValueError(
            "Training split classes do not match baseline checkpoint classes."
        )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT

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
        train_file, class_to_idx, train_transform
    )

    val_dataset = PlantDataset(
        val_file, class_to_idx, eval_transform
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

    model = models.mobilenet_v3_small(weights=None)

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        input_features, len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    # Freeze every feature layer first.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Unfreeze only feature layers 9, 10, 11 and 12.
    for layer in model.features[UNFREEZE_FROM:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Keep the complete classifier trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    feature_parameters = [
        parameter
        for parameter in model.features.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {
            "params": feature_parameters,
            "lr": FEATURE_LR
        },
        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR
        }
    ])

    criterion = nn.CrossEntropyLoss()

    print("\n===== SENTINEL FINE-TUNING =====")
    print(f"Device: {DEVICE}")
    print(f"Classes: {len(class_names)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Feature layers: {len(model.features)}")
    print(f"Frozen feature layers: 0 to {UNFREEZE_FROM - 1}")
    print(f"Trainable feature layers: {UNFREEZE_FROM} to 12")
    print(f"Feature learning rate: {FEATURE_LR}")
    print(f"Classifier learning rate: {CLASSIFIER_LR}")
    print(f"Epochs: {EPOCHS}")
    print("Baseline checkpoint loaded successfully.")

    best_accuracy = 0.0
    history = []

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()

        # Keep frozen early layers and their BatchNorm statistics fixed.
        for layer in model.features[:UNFREEZE_FROM]:
            layer.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

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
                "baseline_validation_accuracy": checkpoint[
                    "validation_accuracy"
                ]
            }, OUTPUT_PATH)

            print("Fine-tuned checkpoint saved.", flush=True)

        with open(
            REPORT_DIR / "finetune_history.csv",
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

    print("\n===== FINE-TUNING COMPLETE =====")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Original baseline validation accuracy: "
          f"{checkpoint['validation_accuracy']:.4f}")
    print(f"Fine-tuned checkpoint: {OUTPUT_PATH}")
    print("Validation history: reports/finetune_history.csv")


if __name__ == "__main__":
    main()

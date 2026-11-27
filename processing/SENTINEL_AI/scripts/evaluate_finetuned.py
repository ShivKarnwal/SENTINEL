
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/raw/PlantVillage"
SPLIT_DIR = ROOT / "datasets/splits/grouped"
MODEL_PATH = ROOT / "models/sentinel_mobilenetv3_small_finetuned_best.pth"
REPORT_DIR = ROOT / "reports"

BATCH_SIZE = 8
NUM_WORKERS = 0

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

                class_name = relative_path.split("/")[2]

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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("Loading fine-tuned checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    test_dataset = PlantDataset(
        SPLIT_DIR / "test.txt",
        class_to_idx,
        transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = models.mobilenet_v3_small(weights=None)

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        input_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    actual_labels = []
    predicted_labels = []

    print(f"Testing images: {len(test_dataset)}")
    print("Evaluating model...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(1)

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    accuracy = accuracy_score(
        actual_labels, predicted_labels
    )

    macro_precision = precision_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    print("\n===== FINE-TUNED TEST RESULTS =====")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['validation_accuracy']:.4f}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_path = REPORT_DIR / "finetuned_test_classification_report.csv"

    with open(report_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()

        for class_name in class_names:
            row = report[class_name]
            writer.writerow({
                "class": class_name,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1-score": row["f1-score"],
                "support": row["support"]
            })

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names)))
    )

    matrix_path = REPORT_DIR / "finetuned_test_confusion_matrix.csv"

    with open(matrix_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + class_names)

        for index, row in enumerate(matrix):
            writer.writerow([class_names[index]] + row.tolist())

    print("\nReports saved:")
    print(report_path)
    print(matrix_path)
    print("\nFINE-TUNED TEST EVALUATION COMPLETE")


if __name__ == "__main__":
    main()

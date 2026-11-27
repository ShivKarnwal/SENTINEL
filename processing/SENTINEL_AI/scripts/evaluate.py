
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/raw/PlantVillage"
SPLIT_FILE = ROOT / "datasets/splits/grouped/test.txt"
CHECKPOINT = ROOT / "models/sentinel_mobilenetv3_small_best.pth"
CLASS_FILE = ROOT / "reports/class_mapping.json"
REPORT_DIR = ROOT / "reports"

BATCH_SIZE = 8
DEVICE = torch.device("cpu")


class PlantTestDataset(Dataset):
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

    with open(CLASS_FILE, "r") as file:
        class_to_idx = json.load(file)

    class_names = [
        name for name, index in
        sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    test_dataset = PlantTestDataset(
        SPLIT_FILE, class_to_idx, transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names)
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    number_of_classes = len(class_names)
    confusion = torch.zeros(
        number_of_classes,
        number_of_classes,
        dtype=torch.int64
    )

    total = 0
    correct = 0

    print("SENTINEL FINAL TEST EVALUATION")
    print("=" * 45)
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy: "
          f"{checkpoint['validation_accuracy']:.4f}")
    print(f"Testing images: {len(test_dataset)}")
    print("Evaluating...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            indices = labels * number_of_classes + predictions
            batch_confusion = torch.bincount(
                indices,
                minlength=number_of_classes ** 2
            ).reshape(number_of_classes, number_of_classes)

            confusion += batch_confusion

    accuracy = correct / total

    rows = []
    precisions = []
    recalls = []
    f1_scores = []

    for index, class_name in enumerate(class_names):
        true_positive = confusion[index, index].item()
        false_positive = confusion[:, index].sum().item() - true_positive
        false_negative = confusion[index, :].sum().item() - true_positive
        support = confusion[index, :].sum().item()

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        rows.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support
        })

    macro_precision = sum(precisions) / number_of_classes
    macro_recall = sum(recalls) / number_of_classes
    macro_f1 = sum(f1_scores) / number_of_classes

    with open(REPORT_DIR / "test_classification_report.csv",
              "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(REPORT_DIR / "test_confusion_matrix.csv",
              "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + class_names)

        for index, class_name in enumerate(class_names):
            writer.writerow(
                [class_name] + confusion[index].tolist()
            )

    print("\nFINAL TEST RESULTS")
    print("=" * 45)
    print(f"Correct predictions: {correct}/{total}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nReports saved:")
    print("reports/test_classification_report.csv")
    print("reports/test_confusion_matrix.csv")

    print("\nTEST EVALUATION COMPLETE")


if __name__ == "__main__":
    main()

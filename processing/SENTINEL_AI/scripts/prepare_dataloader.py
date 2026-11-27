from pathlib import Path
from collections import Counter
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path("datasets/raw/PlantVillage")
SPLITS = Path("datasets/splits/grouped")
IMAGE_SIZE = 224
BATCH_SIZE = 8

# Use the normalization values expected by pretrained MobileNetV3.
from torchvision.models import MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

evaluation_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


class PlantVillageDataset(Dataset):
    def __init__(self, split_file, class_to_idx, transform):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        with open(split_file, "r") as file:
            paths = [line.strip() for line in file if line.strip()]

        for relative_path in paths:
            parts = Path(relative_path).parts

            if len(parts) < 4:
                raise ValueError(f"Unexpected image path: {relative_path}")

            class_name = parts[2]

            if class_name not in class_to_idx:
                raise ValueError(f"Unknown class: {class_name}")

            image_path = ROOT / relative_path

            if not image_path.is_file():
                raise FileNotFoundError(f"Image missing: {image_path}")

            self.samples.append(
                (image_path, class_to_idx[class_name])
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
    print("\nSENTINEL DATA LOADER CHECK")
    print("=" * 45)

    train_file = SPLITS / "train.txt"
    validation_file = SPLITS / "validation.txt"
    test_file = SPLITS / "test.txt"

    for file_path in [train_file, validation_file, test_file]:
        if not file_path.is_file():
            raise FileNotFoundError(f"Split file missing: {file_path}")

    with open(train_file, "r") as file:
        train_paths = [line.strip() for line in file if line.strip()]

    classes = sorted({
        Path(path).parts[2]
        for path in train_paths
    })

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    if len(classes) != 38:
        raise ValueError(
            f"Expected 38 classes, but found {len(classes)}"
        )

    with open("models/class_to_idx.json", "w") as file:
        json.dump(class_to_idx, file, indent=4)

    train_dataset = PlantVillageDataset(
        train_file, class_to_idx, train_transform
    )
    validation_dataset = PlantVillageDataset(
        validation_file, class_to_idx, evaluation_transform
    )
    test_dataset = PlantVillageDataset(
        test_file, class_to_idx, evaluation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Classes: {len(classes)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    images, labels = next(iter(train_loader))

    print("\nFIRST TRAINING BATCH")
    print(f"Image tensor shape: {tuple(images.shape)}")
    print(f"Label tensor shape: {tuple(labels.shape)}")
    print(f"Image data type: {images.dtype}")
    print(f"Label data type: {labels.dtype}")
    print(f"Minimum label: {labels.min().item()}")
    print(f"Maximum label: {labels.max().item()}")

    print("\nDATA LOADER CHECK COMPLETE")


if __name__ == "__main__":
    main()

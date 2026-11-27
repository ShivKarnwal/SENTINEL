from pathlib import Path
import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/raw/PlantVillage"
NO_PLANT_DIR = DATA_ROOT / "raw/color/No_Plant"
SPLIT_FILE = ROOT / "datasets/splits/grouped/train.txt"

BASELINE = ROOT / "models/sentinel_mobilenetv3_small_finetuned_best.pth"
OUTPUT = ROOT / "models/sentinel_mobilenetv3_small_39class_best.pth"

EPOCHS = 30
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

checkpoint = torch.load(BASELINE, map_location="cpu", weights_only=False)
old_mapping = checkpoint["class_to_idx"]

if len(old_mapping) != 38:
    raise ValueError("Expected original 38-class checkpoint.")

no_plant_images = sorted(
    p for p in NO_PLANT_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(no_plant_images) != 4:
    raise ValueError(f"Expected 4 images, found {len(no_plant_images)}")

if not SPLIT_FILE.exists():
    raise FileNotFoundError(SPLIT_FILE)

plant_images = []
for line in SPLIT_FILE.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    path = DATA_ROOT / line
    if path.exists():
        plant_images.append(path)

if not plant_images:
    raise RuntimeError("No original plant training images found.")

class_to_idx = dict(old_mapping)
class_to_idx["No_Plant"] = 38

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class ImageDataset(Dataset):
    def __init__(self, paths, label):
        self.paths = paths
        self.label = label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = transform(im.convert("RGB"))
        return image, self.label[index]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(1024, 38)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

# Preserve the original 38 outputs.
for parameter in model.parameters():
    parameter.requires_grad = False

# Train a separate 39th output neuron.
new_weight = nn.Parameter(torch.empty(1, 1024))
new_bias = nn.Parameter(torch.zeros(1))
nn.init.normal_(new_weight, mean=0.0, std=0.01)

optimizer = torch.optim.Adam(
    [new_weight, new_bias],
    lr=0.0001
)

criterion = nn.BCEWithLogitsLoss()

print("SENTINEL 39-CLASS EXPERIMENT")
print("Original classes:", len(old_mapping))
print("No_Plant images:", len(no_plant_images))
print("Available plant images:", len(plant_images))
print("Training only the new output neuron.")
print("Original checkpoint will not be modified.")

for epoch in range(EPOCHS):
    # Four positive examples and four randomly selected plant examples.
    selected_plants = random.sample(
        plant_images, min(4, len(plant_images))
    )

    paths = no_plant_images + selected_plants
    labels = [1.0] * len(no_plant_images) + [0.0] * len(selected_plants)

    dataset = ImageDataset(paths, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().to(device)

        with torch.no_grad():
            features = model.features(images)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            hidden = model.classifier[0](features)
            hidden = model.classifier[1](hidden)
            hidden = model.classifier[2](hidden)

        new_logits = F.linear(hidden, new_weight, new_bias).squeeze(1)

        optimizer.zero_grad()
        loss = criterion(new_logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {total_loss / len(loader):.5f}"
    )

# Construct a new 39-output classifier.
new_model = models.mobilenet_v3_small(weights=None)
new_model.classifier[3] = nn.Linear(1024, 39)

state = copy.deepcopy(model.state_dict())
state["classifier.3.weight"] = torch.cat(
    [state["classifier.3.weight"], new_weight.detach()], dim=0
)
state["classifier.3.bias"] = torch.cat(
    [state["classifier.3.bias"], new_bias.detach()], dim=0
)

new_model.load_state_dict(state)

torch.save({
    "architecture": "MobileNetV3-Small",
    "model_state_dict": new_model.state_dict(),
    "class_to_idx": class_to_idx,
    "epoch": EPOCHS,
    "validation_accuracy": None,
    "baseline_validation_accuracy": checkpoint["validation_accuracy"],
    "experimental_no_plant_images": len(no_plant_images),
    "experimental_warning": (
        "Only four No_Plant images. Not independently validated. "
        "Do not use as a reliable general-purpose plant detector."
    )
}, OUTPUT)

print("\nEXPERIMENT COMPLETE")
print("Saved:", OUTPUT)
print("Original checkpoint preserved:", BASELINE.exists())

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models/sentinel_mobilenetv3_small_finetuned_best.pth"
EXPERIMENT = ROOT / "models/sentinel_mobilenetv3_small_39class_best.pth"
TEST_FILE = ROOT / "datasets/splits/grouped/test.txt"
NO_PLANT_DIR = ROOT / "datasets/raw/PlantVillage/raw/color/No_Plant"

device = torch.device("cpu")
weights = models.MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

base = torch.load(BASELINE, map_location="cpu", weights_only=False)
exp = torch.load(EXPERIMENT, map_location="cpu", weights_only=False)

old_mapping = base["class_to_idx"]
new_mapping = exp["class_to_idx"]

assert len(old_mapping) == 38
assert len(new_mapping) == 39
assert new_mapping["No_Plant"] == 38

def make_model(classes, state):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, classes)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

original_model = make_model(38, base["model_state_dict"])
experimental_model = make_model(39, exp["model_state_dict"])

class ImageDataset(Dataset):
    def __init__(self, paths, labels=None):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = transform(im.convert("RGB"))
        label = -1 if self.labels is None else self.labels[index]
        return image, label

test_paths = []
test_labels = []

for line in TEST_FILE.read_text().splitlines():
    line = line.strip()
    if not line:
        continue

    relative = Path(line)
    class_name = relative.parts[2]

    if class_name not in old_mapping:
        continue

    path = ROOT / "datasets/raw/PlantVillage" / relative
    if path.exists():
        test_paths.append(path)
        test_labels.append(old_mapping[class_name])

no_plant_paths = sorted(
    p for p in NO_PLANT_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print("\nSENTINEL 39-CLASS VERIFICATION")
print("==============================")
print("Plant test images:", len(test_paths))
print("No_Plant images:", len(no_plant_paths))
print("Device:", device)

loader = DataLoader(
    ImageDataset(test_paths, test_labels),
    batch_size=32,
    shuffle=False,
    num_workers=0
)

old_correct = 0
new_correct = 0
plant_rejected = 0
total = 0

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        old_logits = original_model(images)
        new_logits = experimental_model(images)

        old_predictions = old_logits.argmax(dim=1)
        new_predictions = new_logits.argmax(dim=1)

        old_correct += (old_predictions == labels).sum().item()
        new_correct += (new_predictions == labels).sum().item()
        plant_rejected += (new_predictions == 38).sum().item()
        total += labels.size(0)

print("\nHELD-OUT PLANT TEST")
print("-------------------")
print(f"Original model accuracy: {100 * old_correct / total:.2f}%")
print(f"39-class model plant accuracy: {100 * new_correct / total:.2f}%")
print(f"Plant images incorrectly called No_Plant: {plant_rejected}/{total}")

print("\nYOUR FOUR No_Plant PHOTOS")
print("-------------------------")

no_plant_loader = DataLoader(
    ImageDataset(no_plant_paths),
    batch_size=4,
    shuffle=False
)

with torch.inference_mode():
    for images, _ in no_plant_loader:
        logits = experimental_model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)

        for i, path in enumerate(no_plant_paths):
            top_probability, top_index = probabilities[i].max(dim=0)
            predicted_name = (
                "No_Plant" if top_index.item() == 38
                else next(
                    name for name, index in old_mapping.items()
                    if index == top_index.item()
                )
            )
            no_plant_probability = probabilities[i, 38].item()

            print(f"\nImage: {path.name}")
            print(f"Prediction: {predicted_name}")
            print(f"Top confidence: {top_probability.item() * 100:.2f}%")
            print(f"No_Plant probability: {no_plant_probability * 100:.2f}%")

print("\nVERIFICATION COMPLETE")

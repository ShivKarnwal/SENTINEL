
from pathlib import Path
import json

BASE = Path.cwd()

DATA_DIR = BASE / "datasets/raw/PlantVillage"
META_DIR = BASE / "datasets/raw/PlantVillage_metadata"

TRAIN_FILE = META_DIR / "splits/color_train.txt"
TEST_FILE = META_DIR / "splits/color_test.txt"
LEAF_MAP_FILE = META_DIR / "leaf_grouping/leaf-map.json"


def read_split(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


print("=" * 55)
print("SENTINEL — OFFICIAL SPLIT VERIFICATION")
print("=" * 55)

train_paths = read_split(TRAIN_FILE)
test_paths = read_split(TEST_FILE)

train_set = set(train_paths)
test_set = set(test_paths)

print("\n1. SPLIT COUNTS")
print("Training entries:", len(train_paths))
print("Testing entries:", len(test_paths))
print("Unique training paths:", len(train_set))
print("Unique testing paths:", len(test_set))

print("\n2. DUPLICATE CHECK")
print("Duplicate training entries:", len(train_paths) - len(train_set))
print("Duplicate testing entries:", len(test_paths) - len(test_set))

print("\n3. TRAIN/TEST OVERLAP")
overlap = train_set.intersection(test_set)
print("Paths appearing in both:", len(overlap))

print("\n4. FILE EXISTENCE CHECK")

missing_train = [
    path for path in train_paths
    if not (DATA_DIR / path).is_file()
]

missing_test = [
    path for path in test_paths
    if not (DATA_DIR / path).is_file()
]

print("Missing training images:", len(missing_train))
print("Missing testing images:", len(missing_test))

if missing_train:
    print("Example missing training path:", missing_train[0])

if missing_test:
    print("Example missing testing path:", missing_test[0])

print("\n5. CLASS DISTRIBUTION")

def get_class(path):
    parts = Path(path).parts
    return parts[2] if len(parts) >= 3 else "UNKNOWN"

train_classes = {}
test_classes = {}

for path in train_paths:
    label = get_class(path)
    train_classes[label] = train_classes.get(label, 0) + 1

for path in test_paths:
    label = get_class(path)
    test_classes[label] = test_classes.get(label, 0) + 1

print("Training classes:", len(train_classes))
print("Testing classes:", len(test_classes))

print("\nFirst 10 training class counts:")
for label, count in sorted(train_classes.items())[:10]:
    print(f"{label}: {count}")

print("\n6. LEAF MAP STRUCTURE")

with LEAF_MAP_FILE.open("r", encoding="utf-8") as file:
    leaf_map = json.load(file)

print("JSON top-level type:", type(leaf_map).__name__)

if isinstance(leaf_map, dict):
    print("Top-level keys:", list(leaf_map.keys())[:15])

    for key, value in list(leaf_map.items())[:3]:
        print(f"\nExample key: {key}")
        print("Value type:", type(value).__name__)
        print("Example value:", str(value)[:500])

elif isinstance(leaf_map, list):
    print("Number of entries:", len(leaf_map))
    print("First 3 entries:", leaf_map[:3])

print("\n7. VERIFICATION SUMMARY")

checks = {
    "No duplicate training paths": len(train_paths) == len(train_set),
    "No duplicate testing paths": len(test_paths) == len(test_set),
    "No train/test overlap": len(overlap) == 0,
    "All training images exist": len(missing_train) == 0,
    "All testing images exist": len(missing_test) == 0,
}

for description, passed in checks.items():
    print(("PASS" if passed else "CHECK") + ": " + description)

print("\nVERIFICATION FINISHED")

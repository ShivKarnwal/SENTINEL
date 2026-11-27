
from pathlib import Path
import json

BASE = Path.cwd()
META = BASE / "datasets/raw/PlantVillage_metadata"

with open(META / "leaf_grouping/leaf-map.json", encoding="utf-8") as f:
    leaf_map = json.load(f)


def read_paths(filename):
    with open(META / "splits" / filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def identify(path):
    parts = path.split("/")
    class_name = parts[2]
    file_name = parts[3]

    # Match the original Hugging Face loader
    identifier = file_name.replace("_final_masked", "")

    if "___" in identifier:
        identifier = identifier.split("___")[-1]

    identifier = identifier.split("copy")[0]
    identifier = (
        identifier.replace(".jpg", "")
        .replace(".JPG", "")
        .replace(".png", "")
        .replace(".PNG", "")
        .strip()
    )

    key = identifier.lower().strip()

    if key in leaf_map:
        suggestions = leaf_map[key]

        if len(suggestions) == 1:
            return suggestions[0], "mapped"

        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion, "mapped"

    return f"fallback_{identifier}", "fallback"


train = read_paths("color_train.txt")
test = read_paths("color_test.txt")

train_groups = {}
test_groups = {}
train_types = {}
test_types = {}

for path in train:
    leaf_id, kind = identify(path)
    train_groups.setdefault(leaf_id, []).append(path)
    train_types[leaf_id] = kind

for path in test:
    leaf_id, kind = identify(path)
    test_groups.setdefault(leaf_id, []).append(path)
    test_types[leaf_id] = kind

overlap = set(train_groups) & set(test_groups)

mapped = [
    x for x in overlap
    if train_types[x] == "mapped" and test_types[x] == "mapped"
]

fallback = [
    x for x in overlap
    if train_types[x] == "fallback" or test_types[x] == "fallback"
]

print("=" * 55)
print("SENTINEL — ORIGINAL LOADER LEAF AUDIT")
print("=" * 55)

print("\nIMAGE COUNTS")
print("Training:", len(train))
print("Testing:", len(test))

print("\nOVERLAP COUNTS")
print("Total shared IDs:", len(overlap))
print("Mapped on both sides:", len(mapped))
print("Fallback involved:", len(fallback))

print("\nMAPPED OVERLAP EXAMPLES")
for item in mapped[:5]:
    print("\nID:", item)
    print("TRAIN:", train_groups[item][0])
    print("TEST:", test_groups[item][0])

print("\nFALLBACK OVERLAP EXAMPLES")
for item in fallback[:10]:
    print("\nID:", item)
    print("TRAIN:", train_groups[item][0])
    print("TEST:", test_groups[item][0])

print("\nAUDIT COMPLETE")


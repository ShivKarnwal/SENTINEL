
from pathlib import Path
from collections import defaultdict
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

    # Reproduce the original loader's normalization
    identifier = file_name.replace("_final_masked", "")

    if "___" in identifier:
        identifier = identifier.split("___")[-1]

    identifier = identifier.split("copy")[0]
    identifier = (
        identifier.replace(".jpg", "")
        .replace(".JPG", "")
        .replace(".png", "")
        .replace(".PNG", "")
    ).strip()

    lookup_key = identifier.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]

        if len(suggestions) == 1:
            return suggestions[0], "mapped", class_name

        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion, "mapped", class_name

    return f"fallback_{identifier}", "fallback", class_name


train = read_paths("color_train.txt")
test = read_paths("color_test.txt")

train_groups = defaultdict(list)
test_groups = defaultdict(list)

for path in train:
    leaf_id, status, class_name = identify(path)
    if status == "fallback":
        train_groups[(class_name, leaf_id)].append(path)

for path in test:
    leaf_id, status, class_name = identify(path)
    if status == "fallback":
        test_groups[(class_name, leaf_id)].append(path)


shared_keys = set(train_groups) & set(test_groups)

cross_class_ids = defaultdict(set)

for class_name, leaf_id in shared_keys:
    cross_class_ids[leaf_id].add(class_name)

cross_class_collisions = {
    leaf_id: classes
    for leaf_id, classes in cross_class_ids.items()
    if len(classes) > 1
}

print("=" * 60)
print("SENTINEL — FALLBACK GROUP AUDIT BY CLASS")
print("=" * 60)

print("\nFALLBACK OVERLAP SUMMARY")
print("Shared fallback groups within the same class:", len(shared_keys))
print("Identifiers shared across multiple classes:", len(cross_class_collisions))

print("\nAFFECTED IMAGE COUNTS")
print("Training images in shared groups:",
      sum(len(train_groups[key]) for key in shared_keys))
print("Testing images in shared groups:",
      sum(len(test_groups[key]) for key in shared_keys))

print("\nSHARED GROUPS BY CLASS")

class_counts = defaultdict(int)

for class_name, leaf_id in shared_keys:
    class_counts[class_name] += 1

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nEXAMPLES")

for class_name, leaf_id in sorted(shared_keys)[:10]:
    print("\nClass:", class_name)
    print("Identifier:", leaf_id)
    print("Training images:", len(train_groups[(class_name, leaf_id)]))
    print("Testing images:", len(test_groups[(class_name, leaf_id)]))
    print("TRAIN:", train_groups[(class_name, leaf_id)][0])
    print("TEST :", test_groups[(class_name, leaf_id)][0])

print("\nCROSS-CLASS COLLISION EXAMPLES")

for leaf_id, classes in list(cross_class_collisions.items())[:5]:
    print(leaf_id, "->", sorted(classes))

print("\nAUDIT COMPLETE")

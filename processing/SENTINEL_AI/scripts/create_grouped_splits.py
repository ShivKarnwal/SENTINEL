
from pathlib import Path
from collections import defaultdict
import json
import random
import csv

BASE = Path.cwd()
META = BASE / "datasets/raw/PlantVillage_metadata"
OUTPUT = BASE / "datasets/splits/grouped"

SEED = 42
RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

random.seed(SEED)

with open(META / "leaf_grouping/leaf-map.json", encoding="utf-8") as f:
    leaf_map = json.load(f)


def read_paths(filename):
    with open(META / "splits" / filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def identify(path):
    parts = path.split("/")
    class_name = parts[2]
    file_name = parts[3]

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
            return suggestions[0], class_name

        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion, class_name

    return f"fallback_{identifier}", class_name


all_paths = (
    read_paths("color_train.txt")
    + read_paths("color_test.txt")
)

if len(all_paths) != 54305:
    raise RuntimeError(
        f"Expected 54305 images, found {len(all_paths)}"
    )

if len(set(all_paths)) != len(all_paths):
    raise RuntimeError("Duplicate image paths detected.")

# Group images by class and leaf identifier.
groups_by_class = defaultdict(lambda: defaultdict(list))

for path in all_paths:
    leaf_id, class_name = identify(path)
    groups_by_class[class_name][leaf_id].append(path)


splits = {
    "train": [],
    "validation": [],
    "test": [],
}

manifest = []

for class_name, groups in sorted(groups_by_class.items()):
    group_items = list(groups.items())

    # Randomize ties, then process larger groups first.
    random.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    class_total = sum(len(paths) for _, paths in group_items)

    targets = {
        name: class_total * ratio
        for name, ratio in RATIOS.items()
    }

    counts = {name: 0 for name in RATIOS}

    for leaf_id, paths in group_items:
        size = len(paths)

        # Assign the entire group to the split with
        # the largest relative remaining deficit.
        def score(name):
            target = max(targets[name], 1)
            return (targets[name] - counts[name]) / target

        selected = max(RATIOS, key=score)

        splits[selected].extend(paths)
        counts[selected] += size

        for path in paths:
            manifest.append({
                "class": class_name,
                "leaf_id": leaf_id,
                "split": selected,
                "image_path": path,
            })

    print(
        f"{class_name}: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}"
    )


OUTPUT.mkdir(parents=True, exist_ok=True)

for split_name, paths in splits.items():
    random.shuffle(paths)

    with open(
        OUTPUT / f"{split_name}.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(paths) + "\n")


with open(
    OUTPUT / "group_manifest.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class", "leaf_id", "split", "image_path"]
    )
    writer.writeheader()
    writer.writerows(manifest)


# Final verification: no group can occur in multiple splits.
group_splits = defaultdict(set)

for row in manifest:
    key = (row["class"], row["leaf_id"])
    group_splits[key].add(row["split"])

leaking_groups = [
    key for key, assigned in group_splits.items()
    if len(assigned) > 1
]

print("\n" + "=" * 55)
print("SENTINEL — GROUPED DATASET SPLIT")
print("=" * 55)

for name, paths in splits.items():
    print(f"{name}: {len(paths)} images")

print("Total:", sum(len(paths) for paths in splits.values()))
print("Unique leaf groups:", len(group_splits))
print("Groups crossing splits:", len(leaking_groups))

if leaking_groups:
    raise RuntimeError("LEAKAGE CHECK FAILED")

print("\nOriginal metadata splits were not modified.")
print("New files saved in:", OUTPUT)
print("GROUPED SPLIT COMPLETE")

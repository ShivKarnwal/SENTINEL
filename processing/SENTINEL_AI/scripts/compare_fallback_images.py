from pathlib import Path
from collections import defaultdict
from PIL import Image
import json
import csv

BASE = Path.cwd()
META = BASE / "datasets/raw/PlantVillage_metadata"
IMAGE_ROOT = BASE / "datasets/raw/PlantVillage"
OUTPUT = BASE / "reports"

OUTPUT.mkdir(exist_ok=True)

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
            return suggestions[0], "mapped", class_name

        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion, "mapped", class_name

    return f"fallback_{identifier}", "fallback", class_name


def difference_hash(path):
    image_path = IMAGE_ROOT / path

    with Image.open(image_path) as image:
        image = image.convert("L").resize((9, 8))

    pixels = list(image.getdata())
    bits = 0

    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]

            bits = (bits << 1) | int(left > right)

    return bits


def hamming_distance(hash_a, hash_b):
    return (hash_a ^ hash_b).bit_count()


train_groups = defaultdict(list)
test_groups = defaultdict(list)

for path in read_paths("color_train.txt"):
    leaf_id, status, class_name = identify(path)

    if status == "fallback":
        train_groups[(class_name, leaf_id)].append(path)

for path in read_paths("color_test.txt"):
    leaf_id, status, class_name = identify(path)

    if status == "fallback":
        test_groups[(class_name, leaf_id)].append(path)


shared_keys = sorted(set(train_groups) & set(test_groups))

results = []

for class_name, leaf_id in shared_keys:
    for train_path in train_groups[(class_name, leaf_id)]:
        train_hash = difference_hash(train_path)

        for test_path in test_groups[(class_name, leaf_id)]:
            test_hash = difference_hash(test_path)

            distance = hamming_distance(train_hash, test_hash)

            results.append({
                "class": class_name,
                "fallback_id": leaf_id,
                "distance": distance,
                "train_path": train_path,
                "test_path": test_path,
            })

results.sort(key=lambda item: item["distance"])

csv_path = OUTPUT / "fallback_similarity_audit.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "fallback_id",
            "distance",
            "train_path",
            "test_path",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("=" * 60)
print("SENTINEL — PERCEPTUAL IMAGE SIMILARITY AUDIT")
print("=" * 60)

print("Shared fallback groups:", len(shared_keys))
print("Image pairs compared:", len(results))

print("\nDISTANCE SUMMARY")
for threshold in [4, 8, 12, 16]:
    count = sum(
        result["distance"] <= threshold
        for result in results
    )
    print(f"Pairs with distance <= {threshold}: {count}")

print("\nMOST SIMILAR PAIRS")
for result in results[:15]:
    print("\nDistance:", result["distance"])
    print("Class:", result["class"])
    print("Identifier:", result["fallback_id"])
    print("TRAIN:", result["train_path"])
    print("TEST :", result["test_path"])

print("\nCSV report:", csv_path)
print("AUDIT COMPLETE")

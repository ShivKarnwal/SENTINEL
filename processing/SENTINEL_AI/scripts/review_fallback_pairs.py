from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import json

BASE = Path.cwd()

META = BASE / "datasets/raw/PlantVillage_metadata"
IMAGE_ROOT = BASE / "datasets/raw/PlantVillage"

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

# Focus on the class with the largest number of shared groups.
target_class = "Corn_(maize)___healthy"

selected = [
    key for key in shared_keys
    if key[0] == target_class
][:12]

if not selected:
    raise RuntimeError("No matching pairs found for the selected class.")


WIDTH = 700
ROW_HEIGHT = 310

sheet = Image.new(
    "RGB",
    (WIDTH, ROW_HEIGHT * len(selected)),
    "white"
)

draw = ImageDraw.Draw(sheet)


def load_thumbnail(relative_path):
    image_path = IMAGE_ROOT / relative_path

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((256, 256))
        return image.copy()


for index, key in enumerate(selected):
    class_name, leaf_id = key

    y = index * ROW_HEIGHT

    draw.text(
        (10, y + 5),
        f"Pair {index + 1} | {leaf_id}",
        fill="black"
    )

    train_path = train_groups[key][0]
    test_path = test_groups[key][0]

    train_image = load_thumbnail(train_path)
    test_image = load_thumbnail(test_path)

    sheet.paste(train_image, (10, y + 35))
    sheet.paste(test_image, (360, y + 35))

    draw.text(
        (10, y + 295),
        "TRAIN",
        fill="black"
    )

    draw.text(
        (360, y + 295),
        "TEST",
        fill="black"
    )


output_dir = BASE / "reports"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "fallback_pair_review.jpg"

sheet.save(output_path, quality=95)

print("=" * 55)
print("SENTINEL — VISUAL FALLBACK PAIR REVIEW")
print("=" * 55)
print("Class:", target_class)
print("Pairs generated:", len(selected))
print("Saved to:", output_path)
print("\nOpen the image and compare each TRAIN/TEST pair.")

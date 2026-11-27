from pathlib import Path
from PIL import Image, ImageDraw
import csv

BASE = Path.cwd()
IMAGE_ROOT = BASE / "datasets/raw/PlantVillage"
CSV_PATH = BASE / "reports/fallback_similarity_audit.csv"
OUTPUT = BASE / "reports/high_similarity_review.jpg"

matches = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if int(row["distance"]) <= 4:
            matches.append(row)

if not matches:
    raise RuntimeError("No pairs found with distance <= 4.")

WIDTH = 750
ROW_HEIGHT = 330

sheet = Image.new(
    "RGB",
    (WIDTH, ROW_HEIGHT * len(matches)),
    "white"
)

draw = ImageDraw.Draw(sheet)


def load_image(relative_path):
    path = IMAGE_ROOT / relative_path

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((280, 260))
        return img.copy()


for index, row in enumerate(matches):
    y = index * ROW_HEIGHT

    draw.text(
        (10, y + 5),
        f"Pair {index + 1} | Distance: {row['distance']}",
        fill="black"
    )

    draw.text(
        (10, y + 25),
        row["fallback_id"][:75],
        fill="black"
    )

    train_image = load_image(row["train_path"])
    test_image = load_image(row["test_path"])

    sheet.paste(train_image, (10, y + 50))
    sheet.paste(test_image, (390, y + 50))

    draw.text((10, y + 315), "TRAIN", fill="black")
    draw.text((390, y + 315), "TEST", fill="black")

sheet.save(OUTPUT, quality=95)

print("Pairs displayed:", len(matches))
print("Saved:", OUTPUT)

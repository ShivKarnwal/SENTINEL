from pathlib import Path
from PIL import Image, UnidentifiedImageError

DATASET = Path("datasets/raw/PlantVillage/raw/color")

valid_images = 0
corrupted_images = 0
dimensions = {}

print("=" * 50)
print("       SENTINEL IMAGE VALIDATION")
print("=" * 50)

image_files = [
    path
    for path in DATASET.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print(f"\nImages found: {len(image_files)}")
print("\nChecking images...")

for index, path in enumerate(image_files, start=1):

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            size = img.size
            dimensions[size] = dimensions.get(size, 0) + 1

        valid_images += 1

    except (OSError, UnidentifiedImageError, ValueError):
        corrupted_images += 1
        print("Corrupted:", path)

    if index % 5000 == 0:
        print(f"Checked {index}/{len(image_files)}")

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

print("Total images:", len(image_files))
print("Valid images:", valid_images)
print("Corrupted images:", corrupted_images)

print("\nMost common image dimensions:")

for size, count in sorted(
    dimensions.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{size}: {count} images")

print("\nVALIDATION COMPLETE")

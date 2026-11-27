from zipfile import ZipFile
from pathlib import PurePosixPath
from collections import Counter

archive_path = "datasets/raw/PlantVillage.zip"

image_extensions = {".jpg", ".jpeg", ".png"}

class_counts = Counter()
total_image_size = 0
total_images = 0

print("=" * 45)
print("       SENTINEL ZIP INSPECTION")
print("=" * 45)

with ZipFile(archive_path, "r") as archive:

    for item in archive.infolist():

        path = PurePosixPath(item.filename)

        if (
            len(path.parts) >= 3
            and path.parts[0] == "raw"
            and path.parts[1] == "color"
            and path.suffix.lower() in image_extensions
        ):
            class_name = path.parts[2]

            class_counts[class_name] += 1
            total_images += 1
            total_image_size += item.file_size

print("\nNUMBER OF CLASSES:", len(class_counts))
print("TOTAL IMAGES:", total_images)

print("\nEXTRACTED IMAGE SIZE:")
print(f"{total_image_size / (1024 ** 3):.2f} GB")

print("\nIMAGES PER CLASS:")
print("-" * 45)

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\n" + "=" * 45)
print("INSPECTION COMPLETE")
print("=" * 45)

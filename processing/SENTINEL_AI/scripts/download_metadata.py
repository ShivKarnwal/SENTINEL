from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

dataset_name = "mohanty/PlantVillage"

destination = Path("datasets/raw/PlantVillage_metadata")
destination.mkdir(parents=True, exist_ok=True)

files_to_download = [
    "README.md",
    "splits/color_train.txt",
    "splits/color_test.txt",
    "leaf_grouping/leaf-map.json"
]

print("=" * 50)
print("SENTINEL METADATA DOWNLOAD")
print("=" * 50)

for filename in files_to_download:

    print(f"\nDownloading: {filename}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename
        )

        target_path = destination / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(downloaded_path, target_path)

        print("Saved to:", target_path)

    except Exception as error:
        print("FAILED:", filename)
        print("Reason:", error)

print("\nMETADATA DOWNLOAD COMPLETE")

from huggingface_hub import HfApi

api = HfApi()

dataset_name = "mohanty/PlantVillage"

info = api.dataset_info(dataset_name)

print("SENTINEL DATASET INSPECTION")
print("===========================")
print("Dataset:", dataset_name)
print("Author:", info.author)
print("Last modified:", info.last_modified)
print("\nDataset files:")

files = api.list_repo_files(
    repo_id=dataset_name,
    repo_type="dataset"
)

for file in files[:40]:
    print(file)

print("\nTotal repository files listed:", len(files))

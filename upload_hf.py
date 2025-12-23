from huggingface_hub import HfApi, Repository
import os

# Repo info
repo_id = "JonWen/digital_twin"
file_path = "data/sentences.csv"
target_path_in_repo = "sentences.csv"  # Where the file will appear in the repo

api = HfApi()

# Upload file
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=target_path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    token=""
)

print("Upload successful.")

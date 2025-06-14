from huggingface_hub import HfApi
import os

api = HfApi()

# Files to upload
files_to_upload = [
    "adapter_config.json",
    "adapter_model.safetensors"
]

print("Uploading LoRA adapter files...")

for filename in files_to_upload:
    if os.path.exists(filename):
        try:
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id="Maorb23/ltxv-news-lora",
                repo_type="model"
            )
            print(f"✅ Successfully uploaded {filename}")
        except Exception as e:
            print(f"❌ Error uploading {filename}: {e}")
    else:
        print(f"❌ File {filename} not found")

print("Upload complete! Your LoRA model should now work with PEFT.")

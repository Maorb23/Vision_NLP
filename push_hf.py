from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# Create the repository first
api.create_repo(
    repo_id="Maorb23/ltxv-news-lora",
    repo_type="model",
    private=False  # Set to True if you want it private
)

# Then upload your model
api.upload_folder(
    folder_path="/workspace/Vision_NLP/outputs/ltxv_lora",
    repo_id="Maorb23/ltxv-news-lora",
    repo_type="model"
)

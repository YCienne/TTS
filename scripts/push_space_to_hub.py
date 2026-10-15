import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

import subprocess
if os.path.isdir("/content/tts"):
    subprocess.run(["git", "-C", "/content/tts", "pull", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "https://github.com/YCienne/TTS.git", "/content/tts"], check=True)

from huggingface_hub import HfApi, create_repo
repo_id = "Lhue-Lhue/akan-twi-tts-demo"
create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

api = HfApi()
api.upload_folder(folder_path="/content/tts/space", repo_id=repo_id, repo_type="space")
print("done:", "https://huggingface.co/spaces/" + repo_id)

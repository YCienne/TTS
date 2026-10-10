import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

from huggingface_hub import HfApi, create_repo
repo_id = "Lhue-Lhue/akan-tts-mms-vits"
create_repo(repo_id, repo_type="model", exist_ok=True)

lines = [
    "---",
    "license: cc-by-nc-4.0",
    "base_model: facebook/mms-tts-aka",
    "language:",
    "- ak",
    "tags:",
    "- text-to-speech",
    "- vits",
    "- akan",
    "- twi",
    "---",
    "",
    "# Akan (Twi) TTS -- fine-tuned MMS VITS",
    "",
    "Fine-tuned from `facebook/mms-tts-aka` (Meta MMS, CC-BY-NC-4.0, non-commercial use only) on",
    "single-speaker Twi speech (WAXAL `twi_tts`, CC-BY-4.0), 100 epochs.",
    "",
    "Built for the University of Ghana HCI Lab TTS task. Code, data prep, evaluation and the full",
    "training recipe: https://github.com/YCienne/TTS",
    "",
    "## Usage",
    "```python",
    "from transformers import AutoTokenizer, VitsModel",
    "import torch, soundfile as sf",
    "",
    "tok = AutoTokenizer.from_pretrained(\"" + repo_id + "\")",
    "model = VitsModel.from_pretrained(\"" + repo_id + "\")",
    "inputs = tok(\"Akwaaba!\", return_tensors=\"pt\")",
    "with torch.no_grad():",
    "    wav = model(**inputs).waveform[0].numpy()",
    "sf.write(\"out.wav\", wav, model.config.sampling_rate)",
    "```",
    "",
    "## Evaluation",
    "Automated round-trip ASR evaluation (CER/WER, vowel-harmony confusions) and a native-speaker",
    "listening test are described in the repository's README and docs folder.",
    "",
    "## Licence",
    "Non-commercial use only (CC-BY-NC-4.0), inherited from the base model.",
]
card = chr(10).join(lines)

with open("/content/model_card.md", "w", encoding="utf-8") as f:
    f.write(card)

api = HfApi()
api.upload_file(path_or_fileobj="/content/model_card.md", path_in_repo="README.md", repo_id=repo_id, repo_type="model")
api.upload_folder(folder_path="/content/drive/MyDrive/akan_tts/runs/speaker2_punct", repo_id=repo_id, repo_type="model")
print("done:", "https://huggingface.co/" + repo_id)

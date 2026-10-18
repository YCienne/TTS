"""Push a fine-tuned checkpoint and its model card to the Hugging Face Hub.

    python scripts/push_to_hub.py --speaker 3 --checkpoint /kaggle/working/akan_tts/runs/speaker3_punct
    python scripts/push_to_hub.py --speaker 2 --checkpoint /content/drive/MyDrive/akan_tts/runs/speaker2_punct

Run this from wherever the checkpoint actually lives -- Kaggle for speaker 3, Colab/Drive for speaker 2
(see README.md's Setup section for which ran where). --checkpoint has a per-speaker, per-platform
default, but pass it explicitly if your run directory differs. The HF token is read, in order: an
already-active huggingface_hub login, a Kaggle secret named HF_TOKEN (via kaggle_secrets), a Colab
secret named HF_TOKEN (via google.colab.userdata), or the HF_TOKEN environment variable.

Each speaker's eval numbers below are copied from docs/report.docx section 3.1
(scripts/eval_synthesis.py, 20 held-out test prompts) -- update them here and re-run if you re-evaluate.
"""
import argparse
import os
from pathlib import Path

EVAL = {
    2: dict(cer=0.211, wer=0.654, confusions=(4, 6, 7, 6), total=23,
            env="Google Colab (single NVIDIA T4 GPU), checkpoints written to Google Drive."),
    3: dict(cer=0.235, wer=0.617, confusions=(2, 0, 2, 3), total=7,
            env="Kaggle (scripts/kaggle_run_all.py), single NVIDIA T4 GPU -- Kaggle's default runtime "
                "offers two T4s, but a dual-GPU run hit an NCCL hang partway through, so the script "
                "forces single-GPU execution (CUDA_VISIBLE_DEVICES=0) to match the Colab-validated "
                "recipe. No Drive mount on Kaggle: output is written to the kernel's own working "
                "directory and pulled down with `kaggle kernels output` after the run completes."),
}
DEFAULT_CHECKPOINT = {
    2: "/content/drive/MyDrive/akan_tts/runs/speaker2_punct",
    3: "/kaggle/working/akan_tts/runs/speaker3_punct",
}


def _hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    raise SystemExit(
        "No HF token found. Log in with `huggingface-cli login`, set the HF_TOKEN env var, or add an "
        "HF_TOKEN secret in Kaggle/Colab."
    )


def build_card(speaker: int, repo_id: str, other_repo_id: str, final: bool) -> str:
    e = EVAL[speaker]
    other_speaker = 3 if speaker == 2 else 2
    role = "the final submitted checkpoint" if final else "a comparison run, not demo-selectable"
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
        f"# Akan (Twi) TTS -- fine-tuned MMS VITS (speaker {speaker})",
        "",
        "Fine-tuned from `facebook/mms-tts-aka` (Meta MMS, CC-BY-NC-4.0, non-commercial use only) on",
        f"single-speaker Twi speech (WAXAL `twi_tts`, speaker {speaker}, CC-BY-4.0), 100 epochs. "
        f"This checkpoint is {role}; see the project README for the full speaker-2/speaker-3 tradeoff.",
        "",
        "Built for the University of Ghana HCI Lab TTS task. Code, data prep, evaluation and the full",
        "training recipe: https://github.com/YCienne/TTS",
        "",
        f"A speaker-{other_speaker} checkpoint from the same recipe is published at",
        f"https://huggingface.co/{other_repo_id}.",
        "",
        "## Computing environment",
        e["env"],
        "",
        "## Usage",
        "```python",
        "from transformers import AutoTokenizer, VitsModel",
        "import torch, soundfile as sf",
        "",
        f'tok = AutoTokenizer.from_pretrained("{repo_id}")',
        f'model = VitsModel.from_pretrained("{repo_id}")',
        'inputs = tok("Ɛyɛ dɛ yie.", return_tensors="pt")',
        "with torch.no_grad():",
        "    wav = model(**inputs).waveform[0].numpy()",
        'sf.write("out.wav", wav, model.config.sampling_rate)',
        "```",
        "",
        "## Evaluation",
        "Automated round-trip ASR evaluation on 20 held-out test prompts (`scripts/eval_synthesis.py`,",
        "transcribed with `facebook/mms-1b-all`):",
        "",
        f"- Mean CER: {e['cer']}",
        f"- Mean WER: {e['wer']}",
        f"- Vowel-harmony confusions (e->ɛ, ɛ->e, o->ɔ, ɔ->o): "
        f"{', '.join(str(c) for c in e['confusions'])} ({e['total']} of 20 prompts)",
        "",
        "A separate F0 (pitch) contour check (`scripts/eval_prosody.py`, same 20 prompts) found "
        "comparable pitch variation across both checkpoints -- see the report for the full comparison, "
        "since this does not by itself indicate naturalness or tonal correctness.",
        "",
        "Native-speaker listening: a structured, scored protocol (naturalness, intelligibility, tone "
        "and vowel-harmony judgments this orthography and ASR cannot measure) was distributed to 3 "
        "raters; results were not returned by the submission deadline. Separately, an informal, "
        "unscored in-person session with the same 3 listeners found the output's variety was Asante "
        "Twi and preferred speaker 3 for audibility, pronunciation and tone over speaker 2 (see the "
        "report for details).",
        "",
        "## Licence",
        "Non-commercial use only (CC-BY-NC-4.0), inherited from the base model.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=int, required=True, choices=[2, 3])
    ap.add_argument("--checkpoint", help="run directory to upload; defaults by speaker/platform")
    ap.add_argument("--final", action="store_true", default=None,
                     help="mark this checkpoint as the final submitted model (default: speaker 3)")
    ap.add_argument("--readme-only", action="store_true",
                     help="only refresh the model card text; skip uploading checkpoint weights "
                          "(use when the weights on the Hub are already correct)")
    args = ap.parse_args()

    checkpoint = None
    if not args.readme_only:
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT[args.speaker]
        if not Path(checkpoint).exists():
            raise SystemExit(f"checkpoint dir not found: {checkpoint} -- pass --checkpoint explicitly, "
                              f"or use --readme-only to skip the weights upload")
    final = args.final if args.final is not None else (args.speaker == 3)

    repo_id = f"Lhue-Lhue/akan-tts-speaker3-vits" if args.speaker == 3 else "Lhue-Lhue/akan-tts-mms-vits"
    other_repo_id = "Lhue-Lhue/akan-tts-mms-vits" if args.speaker == 3 else "Lhue-Lhue/akan-tts-speaker3-vits"

    os.environ["HF_TOKEN"] = _hf_token()
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", exist_ok=True)
    card = build_card(args.speaker, repo_id, other_repo_id, final)

    api = HfApi()
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                     repo_id=repo_id, repo_type="model")
    if checkpoint:
        api.upload_folder(folder_path=checkpoint, repo_id=repo_id, repo_type="model")
    print("done:", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

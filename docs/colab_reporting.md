# Sharing Colab output through the repository

Instead of copy-pasting long outputs, a notebook can push its log to a separate branch, `colab-logs`, of this
repository, where the reviewer can read it. `main` is never touched.

## One-time setup (about 3 minutes)
1. Create a **fine-grained token**: https://github.com/settings/personal-access-tokens/new
   - Name `colab-logs`, expiration **7 days**, resource owner `YCienne`.
   - Repository access: *Only select repositories*, then `YCienne/TTS`.
   - Repository permissions: **Contents: Read and write** (nothing else).
   - Generate and copy the token (it starts with `github_pat_` and is shown once).
2. In Colab, open **Secrets** (key icon, left sidebar), add a secret named `GITHUB_TOKEN`, paste the token and switch on
   **Notebook access**.
3. Never paste the token into a chat, a cell or a file. When the work is finished, delete it under
   *Settings, Developer settings, Personal access tokens*.

## Use
Run the optional report cell at the end of a notebook, or from any cell:
```python
import os
from google.colab import userdata
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
!python /content/tts/scripts/colab_report.py --name my_run --files /content/train.log
```
Text files keep their last 400 lines (`--tail`). Each report goes to `logs/<timestamp>_<name>/` on `colab-logs`.

## What it protects
The token reaches git through environment variables, so it appears in no command line, remote URL or message. The helper
writes only to `colab-logs`, but a token with Contents write access could push to any branch, which is why it should be
short-lived and scoped to this one repository.

## Connection test (also reports the GPU and what is on Drive)
Paste into any Colab cell. The secret may be named `GITHUB_TOKEN` or `colablogs`.
```python
import os, subprocess
from google.colab import userdata
for secret in ("GITHUB_TOKEN", "colablogs"):
    try:
        os.environ["GITHUB_TOKEN"] = userdata.get(secret); print("using secret:", secret); break
    except Exception as e:
        print(f"secret {secret!r}: {type(e).__name__}")
!test -d /content/tts && git -C /content/tts pull -q || git clone -q https://github.com/YCienne/TTS.git /content/tts
!(echo "== GPU =="; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; echo; echo "== system =="; python --version; free -h | head -2; df -h /content | tail -1; echo; echo "== Drive =="; ls /content/drive/MyDrive/akan_tts 2>&1; ls /content/drive/MyDrive/akan_tts/data 2>&1; ls /content/drive/MyDrive/akan_tts/ckpt 2>&1) > /content/env_report.txt 2>&1
!python /content/tts/scripts/colab_report.py --name connection_test --files /content/env_report.txt
```
It should end with `pushed 1 file(s) to branch colab-logs`.

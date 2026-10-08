"""Push logs and small result files from a Colab run to the `colab-logs` branch of the repository.

    python scripts/colab_report.py --name 05_train --files /content/train.log reports/audit.json

Lets a reviewer read a run's output from the repository instead of copy-pasting it. It only ever
pushes to the branch `colab-logs` (created on first use), never to main. Text files longer than
--tail lines keep their last lines. The token is read from the GITHUB_TOKEN environment variable
(in Colab: Secrets, then os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")). It is passed to git through
environment variables, so it appears in no command line, no remote URL and no printed message.
Commits are made as YCienne.
"""
import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REMOTE = "https://github.com/YCienne/TTS.git"
BRANCH = "colab-logs"
NAME, EMAIL = "YCienne", "155327427+YCienne@users.noreply.github.com"
MAX_BYTES = 2_000_000


def git(repo, *args, env=None, check=True):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if check and p.returncode:
        raise SystemExit(f"git {args[0]} failed:\n{scrub(p.stderr or p.stdout)}")
    return p


def scrub(text: str) -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    return text.replace(token, "***") if token else text


def tail_text(path: Path, n: int) -> bytes:
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
    kept = lines[-n:]
    note = [f"[... {len(lines) - len(kept)} earlier lines omitted ...]"] if len(lines) > n else []
    return ("\n".join(note + kept) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short label for this report, e.g. 05_train")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tail", type=int, default=400, help="keep this many last lines of text files")
    ap.add_argument("--note", default="", help="free text to include with the report")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    remote = os.environ.get("REPORT_REMOTE", REMOTE)  # override is for testing against a local repository
    if not token and remote == REMOTE:
        raise SystemExit("GITHUB_TOKEN is not set (Colab: add it under Secrets, then set os.environ from userdata).")

    env = dict(os.environ)
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraheader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}", GIT_TERMINAL_PROMPT="0")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = f"logs/{stamp}_{args.name}"
    tmp = Path(tempfile.mkdtemp())
    try:
        git(tmp, "init", "-q")
        git(tmp, "remote", "add", "origin", remote)
        if git(tmp, "fetch", "-q", "--depth", "1", "origin", BRANCH, env=env, check=False).returncode == 0:
            git(tmp, "checkout", "-q", "-B", BRANCH, "FETCH_HEAD")
        else:
            git(tmp, "checkout", "-q", "--orphan", BRANCH)
        out = tmp / folder
        out.mkdir(parents=True, exist_ok=True)
        sent, skipped = [], []
        for f in map(Path, args.files):
            if not f.is_file():
                skipped.append(f"{f} (missing)")
                continue
            data = tail_text(f, args.tail) if f.suffix in {".log", ".txt", ".tsv", ".csv", ".json", ".md", ""} \
                else f.read_bytes()
            if len(data) > MAX_BYTES:
                skipped.append(f"{f} (over {MAX_BYTES // 1_000_000} MB)")
                continue
            (out / f.name).write_bytes(data)
            sent.append(f.name)
        (out / "REPORT.md").write_text(
            f"# {args.name}\n\nSent {stamp}.\n\nFiles: {', '.join(sent) or 'none'}\n"
            + (f"Skipped: {', '.join(skipped)}\n" if skipped else "")
            + (f"\n{args.note}\n" if args.note else ""), encoding="utf-8")
        git(tmp, "config", "user.name", NAME)
        git(tmp, "config", "user.email", EMAIL)
        git(tmp, "add", "-A")
        git(tmp, "commit", "-q", "--allow-empty", "-m", f"Colab report: {args.name}")
        git(tmp, "push", "-q", "origin", f"HEAD:{BRANCH}", env=env)
        print(f"pushed {len(sent)} file(s) to branch {BRANCH}: {folder}")
        for s in skipped:
            print("skipped:", s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

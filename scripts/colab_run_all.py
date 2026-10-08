"""Run the whole Colab pipeline in one command, and print one short summary.

    !python /content/tts/scripts/colab_run_all.py --epochs 2

Stages, each skipped when its result already exists: GPU check, pinned environment, training recipe, checkpoint
conversion (cached on Drive), vocabulary extension (cached on Drive), conversion check, dataset, training, and a
sample synthesis. The first failing stage stops the run and prints only its last lines, so there is one thing to
read. The full log goes to Drive. Drive must be mounted first (drive.mount in the notebook cell).
Needs: --epochs 2 for the smoke test, --epochs 100 for the real run, --resume to continue an interrupted run.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

VENV = "/content/venv"
PY = f"{VENV}/bin/python"
REPO = "/content/tts"
RECIPE = "/content/finetune-hf-vits"
DRIVE = "/content/drive/MyDrive/akan_tts"
CK = f"{DRIVE}/ckpt"
PACKAGES = ('torch==2.4.1 transformers==4.44.2 "datasets[audio]==2.21.0" accelerate==0.34.2 pyarrow==17.0.0 '
            '"numpy<2" soundfile librosa Cython setuptools wheel tensorboard matplotlib "pandas<3" scipy')
PROMPTS = ["mehunu mununkum. awia nso rebɔ kɛse pa ara.", "aprɛ yɛ aduaba a ɛyɛ dɛ yie."]
HEARTBEAT_SEC = 90
TAIL_LINES = 40


class Pipeline:
    """Runs shell stages, logs everything to a file, prints one line per stage."""

    def __init__(self, log_path, env=None, tail=TAIL_LINES):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.env = dict(os.environ if env is None else env, MPLBACKEND="Agg")
        self.tail = tail
        self.rows = []  # (name, status, seconds)

    def run(self, name, cmd, cwd=None, skip_if=None):
        if skip_if is not None and skip_if():
            self.rows.append((name, "skipped", 0.0))
            print(f"-  {name}: already done, skipped", flush=True)
            return
        print(f">> {name} ...", flush=True)
        start = last_beat = time.time()
        last = deque(maxlen=self.tail)
        proc = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", executable="/bin/bash"
                                if isinstance(cmd, str) else None)
        for line in proc.stdout:
            self.log.write(line)
            piece = line.replace("\r", "\n").rstrip().splitlines()
            if piece and piece[-1].strip():
                last.append(piece[-1])
            if time.time() - last_beat >= HEARTBEAT_SEC:
                print(f"   ... {name} still running ({(time.time() - start) / 60:.0f} min)", flush=True)
                last_beat = time.time()
        code = proc.wait()
        secs = time.time() - start
        self.log.flush()
        if code:
            self.rows.append((name, f"FAILED (exit {code})", secs))
            print(f"x  {name} FAILED after {secs:.0f}s (exit {code}). Last {len(last)} lines:", flush=True)
            print("\n".join(f"   | {l[:200]}" for l in last), flush=True)
            self.finish(failed=name)
            raise SystemExit(1)
        self.rows.append((name, "ok", secs))
        print(f"ok {name} ({secs:.0f}s)", flush=True)

    def finish(self, failed=None):
        self.log.flush()
        print("\n=== SUMMARY ===")
        for name, status, secs in self.rows:
            print(f"{name:<22} {status:<20} {secs:6.0f}s")
        print(f"full log: {self.log_path}")
        if failed:
            print(f"STOPPED at: {failed}")


def stages(args, p):
    speaker = args.speaker
    run_dir = f"{DRIVE}/runs/speaker{speaker}_{'nopunct' if args.no_punct else 'punct'}"
    model = f"/content/mms-tts-aka-{'train' if args.no_punct else 'punct2'}"
    data = f"{DRIVE}/data/speaker{speaker}"
    ds = "/content/aka_ds"

    p.run("GPU", 'nvidia-smi -L && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
    p.run("pinned environment",
          f"pip install -q uv && rm -rf {VENV} && uv venv {VENV} --python 3.11 -q && "
          f"uv pip install --python {PY} -q {PACKAGES}",
          skip_if=lambda: Path(PY).exists() and subprocess.run(
              [PY, "-c", "import torch, transformers, datasets; assert torch.cuda.is_available()"],
              capture_output=True).returncode == 0)
    p.run("training recipe",
          f"test -d {RECIPE} || git clone -q https://github.com/ylacombe/finetune-hf-vits {RECIPE}; "
          f"cd {RECIPE}/monotonic_align && mkdir -p monotonic_align && {PY} setup.py build_ext --inplace -q "
          f"&& cd {RECIPE} && {PY} -c \"from utils.modeling_vits_training import VitsModelForPreTraining\"",
          skip_if=lambda: bool(list(Path(RECIPE, "monotonic_align", "monotonic_align").glob("*.so"))))
    p.run("convert checkpoint",
          f"if [ -f {CK}/mms-tts-aka-train/config.json ]; then cp -r {CK}/mms-tts-aka-train /content/; else "
          f"cd {RECIPE} && {PY} convert_original_discriminator_checkpoint.py --language_code aka "
          f"--pytorch_dump_folder_path /content/mms-tts-aka-train && mkdir -p {CK} && "
          f"cp -r /content/mms-tts-aka-train {CK}/; fi",
          skip_if=lambda: Path("/content/mms-tts-aka-train/config.json").exists())
    if not args.no_punct:
        p.run("extend vocabulary",
              f"if [ -f {CK}/mms-tts-aka-punct2/config.json ]; then cp -r {CK}/mms-tts-aka-punct2 /content/; else "
              f"{PY} {REPO}/scripts/extend_vocab.py --src /content/mms-tts-aka-train --dst /content/mms-tts-aka-punct2 "
              f"--recipe {RECIPE} && cp -r /content/mms-tts-aka-punct2 {CK}/; fi",
              skip_if=lambda: Path("/content/mms-tts-aka-punct2/config.json").exists())
    p.run("check pretrained weights",
          f"{PY} {REPO}/scripts/check_conversion.py --model {model} --recipe {RECIPE} 2>&1 | grep -v Warning | tail -25; "
          f"test ${{PIPESTATUS[0]}} -eq 0")
    p.run("build dataset",
          f"test -f {data}/metadata.tsv || (echo 'Prepared data not found at {data}: run notebooks 02 and 04' && exit 1); "
          f"rm -rf {ds} && {PY} {REPO}/scripts/make_audiofolder.py --data {data} --out {ds} "
          f"{'--strip-punct' if args.no_punct else ''} && "
          f"{PY} -c \"from datasets import load_dataset; d = load_dataset('{ds}'); "
          f"print({{k: len(v) for k, v in d.items()}}, d['train'][0]['audio']['sampling_rate'])\"")
    if not args.skip_train:
        cfg = json.load(open(f"{REPO}/configs/train_aka.json"))
        cfg.update(dataset_name=ds, model_name_or_path=model, output_dir=run_dir, num_train_epochs=args.epochs,
                   overwrite_output_dir=not args.resume)
        cfg["full_generation_sample_text"] = PROMPTS[0] if not args.no_punct else PROMPTS[0].replace(".", "")
        if args.resume:
            cfg["resume_from_checkpoint"] = "latest"
        Path("/content/run.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        p.run(f"train ({args.epochs} epochs)", f"cd {RECIPE} && {VENV}/bin/accelerate launch run_vits_finetuning.py /content/run.json")
    texts = "\n".join(PROMPTS if not args.no_punct else [t.replace(".", "") for t in PROMPTS])
    Path("/content/prompts.txt").write_text(texts + "\n", encoding="utf-8")
    p.run("synthesize samples",
          f"{PY} {REPO}/scripts/synthesize.py --model {run_dir} --text-file /content/prompts.txt --out /content/samples")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--speaker", default="2")
    ap.add_argument("--no-punct", action="store_true", help="strip . , ? ! and use the unextended vocabulary")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    if not Path("/content/drive/MyDrive").exists():
        raise SystemExit("Drive is not mounted: run  from google.colab import drive; drive.mount('/content/drive')  first")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p = Pipeline(f"{DRIVE}/logs/run_all_{stamp}.log")
    run_dir = stages(args, p)
    p.finish()
    print(f"\nmodel: {run_dir}\nsamples: /content/samples/*.wav   (listen: IPython.display.Audio)")


if __name__ == "__main__":
    main()

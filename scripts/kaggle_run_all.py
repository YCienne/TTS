"""Run the whole fine-tuning pipeline on Kaggle in one command, and print one short summary.

    python scripts/kaggle_run_all.py --epochs 100 --speaker 3

Kaggle variant of colab_run_all.py: no Google Drive mount. Reads the prepared speaker data from a Kaggle
input dataset (/kaggle/input/<slug>/), writes everything else (converted checkpoint, dataset, run output)
under /kaggle/working/akan_tts so it is part of this kernel version's committed output -- pull it down
afterwards with `kaggle kernels output` and sync it to Drive from a machine that already holds Drive
credentials. Never resumes (no checkpoint survives between separate Kaggle kernel runs unless you wire one
up explicitly as a new input dataset), so --resume is not offered here.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = "/kaggle/working/tts"
RECIPE = "/kaggle/working/finetune-hf-vits"
WORK = "/kaggle/working/akan_tts"
CK = f"{WORK}/ckpt"
# Kaggle's image runs Python 3.13, which has no cp313 wheel for torch==2.4.1 (the Colab pin, matched to
# Colab's Python 3.11) or for pyarrow==17.0.0, so those two float to the nearest compatible version.
# transformers is bumped off the Colab pin (4.44.2) too: its tokenizers==0.19.x requirement cannot be
# installed on Python 3.13 at all (wheel or source build -- pyo3-ffi 0.21.2 hard-caps at Python 3.12).
# 4.45.2 is the first transformers release allowing tokenizers>=0.20 (which lifted that pyo3 cap),
# picked as the minimal version bump to limit risk of API drift breaking the recipe's VitsConfig usage.
# torchvision must match the torch build exactly (its compiled ops are ABI-linked to a specific torch
# version) -- Kaggle's preinstalled torchvision is for Kaggle's original (newer) torch, so once torch is
# pinned, torchvision has to be pinned to the matching release or anything that transitively imports it
# (timm, pulled in indirectly) crashes with "operator torchvision::nms does not exist" at import time.
# torchvision 0.20.1 (torch 2.5.1's match) has no cp313 wheel either (same story as torch==2.4.1
# earlier), so both are bumped one more notch to torch==2.6.0 / torchvision==0.21.0 -- the next matched
# pair, and the earliest with cp313 wheels.
# numpy<2 (the Colab pin) turned out to be actively harmful here, not just unnecessary: Kaggle's base
# image ships TensorFlow, JAX, and probably other frameworks built against numpy>=2, and transformers'
# own is_tensor() utility lazily probes each of them in turn (TF, then JAX, ...) when checking a model
# output -- every one of them half-imports against our downgraded numpy and dies on the exact same
# "AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'", so removing TensorFlow alone
# only fixed the first framework in that probe order, not the underlying mismatch. Dropping the pin
# lets numpy resolve to the modern 2.x Kaggle (and transformers==4.45.2, torch==2.6.0) actually expect.
PACKAGES = ('torch==2.6.0 torchvision==0.21.0 transformers==4.45.2 "datasets[audio]==2.21.0" '
            'accelerate==0.34.2 "pyarrow>=18" '
            'soundfile librosa Cython setuptools wheel tensorboard matplotlib "pandas<3" scipy')
PROMPTS = ["mehunu mununkum. awia nso rebɔ kɛse pa ara.", "aprɛ yɛ aduaba a ɛyɛ dɛ yie."]
HEARTBEAT_SEC = 90
# 40 was too short to catch a real training-worker traceback: torchrun's own ChildFailedError summary
# doesn't capture it either (run_vits_finetuning.py isn't wrapped with @record), so the only place the
# actual exception shows up is earlier in the subprocess output, pushed out of a short tail.
TAIL_LINES = 400


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
    run_dir = f"{WORK}/runs/speaker{speaker}_{'nopunct' if args.no_punct else 'punct'}"
    model = f"/kaggle/working/mms-tts-aka-{'train' if args.no_punct else 'punct2'}"
    data = args.data_dir
    ds = "/kaggle/working/aka_ds"

    p.run("GPU", 'nvidia-smi -L && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
    # No skip_if here: Kaggle's base image already has torch+cuda+transformers preinstalled (often a
    # newer transformers than the recipe was written for), so a "does it import and see cuda" check
    # would wrongly skip the pin and leave an incompatible transformers in place.
    # No Rust toolchain needed: tokenizers==0.19.x (pinned transitively by transformers==4.44.2, the
    # Colab pin) has no cp313 wheel on PyPI, and a from-source build was confirmed impossible too --
    # its pyo3-ffi v0.21.2 build dependency hard-caps at Python 3.12 ("the configured Python
    # interpreter version (3.13) is newer than PyO3's maximum supported version"), so no amount of
    # Rust/gcc tooling fixes it. The real fix is bumping transformers to the first release that allows
    # tokenizers>=0.20 (which raised that pyo3 ceiling) -- picking the minimal bump above 4.44.2 to
    # limit the risk of re-hitting the "VitsConfig has no attribute pad_token_id" break that a much
    # newer transformers caused earlier.
    p.run("pinned environment", f"pip install {PACKAGES}")
    # Kaggle's image preinstalls TensorFlow, built against numpy>=2; our numpy<2 pin (needed for the
    # rest of the stack) makes it half-import and crash with "AttributeError: module 'numpy.dtypes'
    # has no attribute 'StringDType'" -- not during our own code, but when torch.utils.tensorboard
    # (used for training logs per train_aka.json's report_to=["tensorboard"]) lazily probes
    # tensorboard.compat.tf for gfile support and ends up importing real TensorFlow. Uninstalling
    # TensorFlow outright makes that probe a clean ImportError instead, which tensorboard already
    # handles gracefully -- no need to touch the numpy pin for the rest of the stack.
    # skip_if must NOT be a plain `import tensorflow` check: tensorflow is broken (same numpy issue)
    # even when still installed, so a bare import raises an exception either way and a check based on
    # that can't tell "removed" from "present but broken" -- it wrongly skipped the real uninstall
    # once already. `pip show` only looks at installed-package metadata, unaffected by whether the
    # package actually imports cleanly.
    p.run("remove conflicting TensorFlow",
          "pip uninstall -y tensorflow tensorflow-cpu tf-keras tensorflow-text tensorflow-io-gcs-filesystem "
          ">/dev/null 2>&1; true",
          skip_if=lambda: subprocess.run(
              ["pip", "show", "tensorflow"], capture_output=True).returncode != 0)
    p.run("training recipe",
          f"test -d {RECIPE} || git clone -q https://github.com/ylacombe/finetune-hf-vits {RECIPE}; "
          f"{sys.executable} {REPO}/scripts/patch_recipe.py --recipe {RECIPE} && "
          f"cd {RECIPE}/monotonic_align && mkdir -p monotonic_align && {sys.executable} setup.py build_ext --inplace -q "
          f"&& cd {RECIPE} && {sys.executable} -c \"from utils.modeling_vits_training import VitsModelForPreTraining\"",
          skip_if=lambda: bool(list(Path(RECIPE, "monotonic_align", "monotonic_align").glob("*.so"))))
    # No Drive cache to check: always convert fresh (the only cache we ever had turned out to be incomplete).
    p.run("convert checkpoint",
          f"cd {RECIPE} && {sys.executable} convert_original_discriminator_checkpoint.py --language_code aka "
          f"--pytorch_dump_folder_path /kaggle/working/mms-tts-aka-train",
          skip_if=lambda: Path("/kaggle/working/mms-tts-aka-train/config.json").exists())
    if not args.no_punct:
        p.run("extend vocabulary",
              f"{sys.executable} {REPO}/scripts/extend_vocab.py --src /kaggle/working/mms-tts-aka-train "
              f"--dst /kaggle/working/mms-tts-aka-punct2 --recipe {RECIPE}",
              skip_if=lambda: Path("/kaggle/working/mms-tts-aka-punct2/config.json").exists())
    p.run("check pretrained weights",
          f"{sys.executable} {REPO}/scripts/check_conversion.py --model {model} --recipe {RECIPE} 2>&1 | grep -v Warning | tail -25; "
          f"test ${{PIPESTATUS[0]}} -eq 0")
    p.run("build dataset",
          f"test -f {data}/metadata.tsv || (echo 'Prepared data not found at {data}' && exit 1); "
          f"rm -rf {ds} && {sys.executable} {REPO}/scripts/make_audiofolder.py --data {data} --out {ds} "
          f"{'--strip-punct' if args.no_punct else ''} && "
          f"{sys.executable} -c \"from datasets import load_dataset; d = load_dataset('{ds}'); "
          f"print({{k: len(v) for k, v in d.items()}}, d['train'][0]['audio']['sampling_rate'])\"")
    if not args.skip_train:
        cfg = json.load(open(f"{REPO}/configs/train_aka.json"))
        cfg.update(dataset_name=ds, model_name_or_path=model, output_dir=run_dir, num_train_epochs=args.epochs,
                   overwrite_output_dir=True)
        cfg["full_generation_sample_text"] = PROMPTS[0] if not args.no_punct else PROMPTS[0].replace(".", "")
        Path("/kaggle/working/run.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        # Single GPU only: a dual-T4 run reached step ~1900/3700 (51%, well past three checkpoint
        # saves) before dying with "c10::DistBackendError: ... Watchdog caught collective operation
        # timeout" during an NCCL ALLGATHER -- a multi-GPU interconnect hang, not a code bug. This
        # recipe was only ever validated single-GPU on Colab anyway; CUDA_VISIBLE_DEVICES=0 makes
        # accelerate skip its multi_gpu_launcher (and NCCL) entirely, trading the second GPU's
        # data-parallel throughput for not risking the same hang again.
        p.run(f"train ({args.epochs} epochs)",
              f"cd {RECIPE} && CUDA_VISIBLE_DEVICES=0 accelerate launch run_vits_finetuning.py /kaggle/working/run.json")
    texts = "\n".join(PROMPTS if not args.no_punct else [t.replace(".", "") for t in PROMPTS])
    Path("/kaggle/working/prompts.txt").write_text(texts + "\n", encoding="utf-8")
    p.run("synthesize samples",
          f"{sys.executable} {REPO}/scripts/synthesize.py --model {run_dir} --text-file /kaggle/working/prompts.txt "
          f"--out /kaggle/working/samples")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--speaker", default="3")
    ap.add_argument("--data-dir", required=True, help="Kaggle input dataset folder with metadata(_filtered).tsv + wavs")
    ap.add_argument("--no-punct", action="store_true", help="strip . , ? ! and use the unextended vocabulary")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    if not Path(args.data_dir).exists():
        raise SystemExit(f"--data-dir not found: {args.data_dir}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p = Pipeline(f"{WORK}/logs/run_all_{stamp}.log")
    run_dir = stages(args, p)
    p.finish()
    print(f"\nmodel: {run_dir}\nsamples: /kaggle/working/samples/*.wav")


if __name__ == "__main__":
    main()

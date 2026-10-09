"""Apply two one-line compatibility fixes to the cloned ylacombe/finetune-hf-vits recipe.

    python scripts/patch_recipe.py --recipe /content/finetune-hf-vits

The recipe is cloned fresh each runtime (never copied into this repo), so this patches it in place right
after cloning. Both bugs were found by actually running the 2-epoch smoke test end to end on our single-
speaker Akan checkpoint; both are upstream issues in the recipe, not in our data or config.

1. speaker_id KeyError. The data collator sets batch["speaker_id"] = None when the dataset has no speaker_id
   column (our case: a single-speaker checkpoint, speaker_embedding_size 0). But the training loop at three
   points reads batch["speaker_id"] with plain dict indexing instead of .get(), and by the time the batch
   reaches those points the key is gone (dropped during tensor/device handling upstream), so Python raises
   KeyError: 'speaker_id' at the very first training step. speaker_id=None is exactly what the model forward
   expects for a single-speaker checkpoint, so batch.get("speaker_id") is the fix: same value, no crash.

2. matplotlib AttributeError. The recipe's utils/plot.py calls fig.canvas.tostring_rgb(), removed from
   FigureCanvasAgg in newer matplotlib (we pin a recent version; the recipe was written for an older one).
   This crashes during the end-of-training alignment plot, after all optimisation steps finish but before the
   checkpoint is saved, so a run can finish training and still lose the checkpoint to this bug.
   fig.canvas.buffer_rgba() is the modern equivalent; it returns 4 channels (RGBA) instead of 3 (RGB), so the
   fix also slices off the alpha channel to keep the function's existing (H, W, 3) contract.

Idempotent: running this twice is a no-op the second time (the strings being replaced are gone after the
first patch).
"""
import argparse
import re
from pathlib import Path


def patch_speaker_id(path):
    text = path.read_text(encoding="utf-8")
    patched = text.replace('speaker_id=batch["speaker_id"]', 'speaker_id=batch.get("speaker_id")')
    n = text.count('speaker_id=batch["speaker_id"]')
    path.write_text(patched, encoding="utf-8")
    return n


def patch_matplotlib(path):
    text = path.read_text(encoding="utf-8")
    n = text.count("fig.canvas.tostring_rgb()")
    text = text.replace("fig.canvas.tostring_rgb()", "fig.canvas.buffer_rgba()")
    text = re.sub(
        r"get_width_height\(\)\[::-1\] \+ \(3,\)\)(?!\[)",
        "get_width_height()[::-1] + (4,))[..., :3]",
        text,
    )
    path.write_text(text, encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, help="clone of ylacombe/finetune-hf-vits")
    args = ap.parse_args()
    recipe = Path(args.recipe)

    n1 = patch_speaker_id(recipe / "run_vits_finetuning.py")
    n2 = patch_matplotlib(recipe / "utils" / "plot.py")
    print(f"patched {n1} speaker_id site(s) in run_vits_finetuning.py, {n2} matplotlib call(s) in utils/plot.py")


if __name__ == "__main__":
    main()

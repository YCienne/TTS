"""Turn prepare_data.py output into a Hugging Face "audiofolder" dataset for the training recipe.

    python scripts/make_audiofolder.py --data <prepare output> --out /content/aka_ds [--strip-punct]

Layout written: <out>/{train,validation,test}/metadata.csv plus the wavs, with the columns
file_name and text. ``load_dataset("<out>")`` then gives an ``audio`` column and a ``text``
column per split. Uses metadata_filtered.tsv when it exists (written by asr_check.py), else
metadata.tsv. --strip-punct removes ". , ? !" for a model whose vocabulary lacks them.
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import strip_punctuation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip-punct", action="store_true")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    meta = data / "metadata_filtered.tsv"
    meta = meta if meta.exists() else data / "metadata.tsv"
    print(f"reading {meta}")
    with open(meta, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    writers, handles, counts, skipped = {}, {}, Counter(), 0
    for r in rows:
        text = strip_punctuation(r["text"]) if args.strip_punct else r["text"]
        if not any(c.isalpha() for c in text):  # empty, or only punctuation
            skipped += 1
            continue
        split = r["split"]
        if split not in writers:
            (out / split).mkdir(parents=True, exist_ok=True)
            handles[split] = open(out / split / "metadata.csv", "w", encoding="utf-8", newline="")
            writers[split] = csv.writer(handles[split])
            writers[split].writerow(["file_name", "text"])
        name = Path(r["file"]).name
        shutil.copy(data / r["file"], out / split / name)
        writers[split].writerow([name, text])
        counts[split] += 1
    for h in handles.values():
        h.close()
    print(dict(counts), f"skipped {skipped} empty")


if __name__ == "__main__":
    main()

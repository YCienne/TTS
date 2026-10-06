"""Audit WAXAL twi_tts for training. Run on Colab after scripts/verify_resources.py.

    python scripts/audit_data.py --out reports/audit.json

Reports, per split and speaker: how many hours survive each duration cap (the raw
clips have a 99 s p90, far above what VITS trains on), what characters remain
outside the MMS vocabulary after normalization, and how many utterances still
contain digits. Also writes a few short clips per speaker to reports/samples/ so a
native speaker can confirm the variety and pick a voice.
"""
import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import find_digits, normalize  # noqa: E402

CAPS = (10, 15, 20, 30)
KEEP_PUNCT = set(".,?!")
SAMPLES_PER_SPEAKER = 3


def main():
    from datasets import Audio, load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="twi_tts")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--out", default="reports/audit.json")
    ap.add_argument("--sample-dir", default="reports/samples")
    args = ap.parse_args()

    allowed = set(AutoTokenizer.from_pretrained("facebook/mms-tts-aka").get_vocab()) | KEEP_PUNCT
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    report, samples = {}, []
    for split in args.splits:
        ds = load_dataset("google/WaxalNLP", args.config, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        spk = defaultdict(lambda: {"dur": [], "gender": ""})
        left_chars, left_utts = Counter(), Counter()
        digits_left = empty = 0
        saved = Counter()
        for row in ds:
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            dur, sid, text = len(wav) / sr, row["speaker_id"], normalize(row["text"])
            spk[sid]["dur"].append(dur)
            spk[sid]["gender"] = row.get("gender") or ""
            empty += not text
            digits_left += find_digits(text)
            bad = [c for c in text if c not in allowed]
            left_chars.update(bad)
            left_utts.update(set(bad))
            if split == "train" and saved[sid] < SAMPLES_PER_SPEAKER and 4 <= dur <= 12:
                path = sample_dir / f"speaker{sid}_{saved[sid]}.wav"
                sf.write(path, wav, sr)
                samples.append({"file": str(path), "speaker": sid, "gender": spk[sid]["gender"],
                                "text": row["text"]})
                saved[sid] += 1
        report[split] = {
            "empty_after_normalization": empty,
            "utterances_with_digits_after_normalization": digits_left,
            "leftover_chars": dict(left_chars.most_common()),
            "utterances_with_leftover_char": dict(left_utts.most_common()),
            "speakers": {
                sid: {
                    "gender": v["gender"],
                    "utterances": len(v["dur"]),
                    "hours": round(sum(v["dur"]) / 3600, 3),
                    "usable": {f"<={c}s": {"utterances": int(sum(d <= c for d in v["dur"])),
                                          "hours": round(sum(d for d in v["dur"] if d <= c) / 3600, 3)}
                               for c in CAPS},
                }
                for sid, v in sorted(spk.items())
            },
        }
    report["samples"] = samples
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

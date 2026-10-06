"""Audit WAXAL twi_tts for training. Run on Colab after scripts/verify_resources.py.

    python scripts/audit_data.py --out reports/audit.json

Reports, per split and speaker: how many hours survive each duration cap (the raw
clips have a 99 s p90, far above what VITS trains on), what characters remain
outside the MMS vocabulary after normalization, and how many utterances still
contain digits. For the train split it also reports recording quality per speaker
(SNR-like level gap, clipping, silence, pitch) with warning flags, and writes a few
short clips per speaker to reports/samples/ so a native speaker can confirm the
variety and pick a voice. Quality numbers rank speakers; listening decides.
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
from akantts.audio_quality import clip_quality, flags, pitch_stats  # noqa: E402
from akantts.normalize import find_digits, normalize  # noqa: E402

CAPS = (10, 15, 20, 30)
KEEP_PUNCT = set(".,?!")
SAMPLES_PER_SPEAKER = 3
PITCH_SECONDS = 20  # pitch tracking is slow, so only the first seconds of a few clips


def _med(xs):
    return round(float(np.median(xs)), 3) if len(xs) else None


def summarize_quality(q, pitch, gender):
    """Collapse per-clip quality dicts into one speaker summary."""
    s = {
        "snr_db_median": _med([c["snr_db"] for c in q]),
        "snr_db_p10": round(float(np.percentile([c["snr_db"] for c in q], 10)), 3),
        "noise_floor_db_median": _med([c["noise_floor_db"] for c in q]),
        "speech_level_db_median": _med([c["speech_level_db"] for c in q]),
        "silence_fraction_median": _med([c["silence_fraction"] for c in q]),
        "clipping_fraction_max": round(max(c["clipping_fraction"] for c in q), 6),
        "clips_with_clipping": int(sum(c["clipping_fraction"] > 1e-3 for c in q)),
        "pitch_clips": len(pitch),
    }
    if pitch:
        meds = [p["f0_median_hz"] for p in pitch]
        s.update({
            "f0_median_hz": _med(meds),
            "f0_clip_median_std_hz": round(float(np.std(meds)), 3),
            "f0_std_semitones_median": _med([p["f0_std_semitones"] for p in pitch]),
            "voiced_fraction_median": _med([p["voiced_fraction"] for p in pitch]),
        })
    s["flags"] = flags(s, gender)
    return s


def main():
    from datasets import Audio, load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="twi_tts")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--out", default="reports/audit.json")
    ap.add_argument("--sample-dir", default="reports/samples")
    ap.add_argument("--pitch-clips", type=int, default=25,
                    help="train clips per speaker used for pitch tracking (0 disables)")
    args = ap.parse_args()

    allowed = set(AutoTokenizer.from_pretrained("facebook/mms-tts-aka").get_vocab()) | KEEP_PUNCT
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    report, samples = {}, []
    for split in args.splits:
        ds = load_dataset("google/WaxalNLP", args.config, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        spk = defaultdict(lambda: {"dur": [], "gender": "", "q": [], "pitch": [], "rate": []})
        left_chars, left_utts = Counter(), Counter()
        digits_left = empty = 0
        saved = Counter()
        for row in ds:
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            dur, sid, text = len(wav) / sr, row["speaker_id"], normalize(row["text"])
            s = spk[sid]
            s["dur"].append(dur)
            s["gender"] = row.get("gender") or ""
            s["q"].append(clip_quality(wav, sr))
            if text:
                s["rate"].append(len(text) / dur)
            if split == "train" and len(s["pitch"]) < args.pitch_clips and 2 <= dur <= 30:
                p = pitch_stats(wav[: PITCH_SECONDS * sr], sr)
                if p:
                    s["pitch"].append(p)
            empty += not text
            digits_left += find_digits(text)
            bad = [c for c in text if c not in allowed]
            left_chars.update(bad)
            left_utts.update(set(bad))
            if split == "train" and saved[sid] < SAMPLES_PER_SPEAKER and 4 <= dur <= 12:
                path = sample_dir / f"speaker{sid}_{saved[sid]}.wav"
                sf.write(path, wav, sr)
                samples.append({"file": str(path), "speaker": sid, "gender": s["gender"],
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
                    "chars_per_sec_median": _med(v["rate"]),
                    "usable": {f"<={c}s": {"utterances": int(sum(d <= c for d in v["dur"])),
                                          "hours": round(sum(d for d in v["dur"] if d <= c) / 3600, 3)}
                               for c in CAPS},
                    "quality": summarize_quality(v["q"], v["pitch"], v["gender"]),
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

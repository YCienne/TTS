"""Build the training set for one WAXAL twi_tts speaker. Run on Colab.

    python scripts/prepare_data.py --speaker 2 --out /content/drive/MyDrive/akan_tts/data

Per clip: short clips (<= --max-sec) are kept whole, trimmed. Longer clips are cut at
sentence pauses (akantts.segment) and kept only if every segment passes a duration and
speaking-rate check; otherwise the whole clip is dropped, because a wrong cut would pair
text with the wrong audio. Text is normalized and rejected if it has characters the model
cannot take (akantts.normalize.to_model_text). Output: 16 kHz mono wavs plus metadata.tsv
(file, split, source_id, text, seconds) and prepare_report.json with every drop reason.
The original WAXAL split of the source clip is kept, so segments of one clip never straddle
train and validation/test.
"""
import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import to_model_text  # noqa: E402
from akantts.segment import plausible, sentences, split_by_pauses  # noqa: E402

TARGET_SR = 16000
KEEP_PUNCT = set(".,?!")


def decode(row):
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sr


def main():
    import librosa
    from datasets import Audio, load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="twi_tts")
    ap.add_argument("--speaker", default="2")
    ap.add_argument("--out", default="data/twi_spk2")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--rate", type=float, default=None,
                    help="speaker median chars/s; default: measured on this speaker's short clips")
    ap.add_argument("--min-pause", type=float, default=0.2)
    args = ap.parse_args()

    allowed = set(AutoTokenizer.from_pretrained("facebook/mms-tts-aka").get_vocab()) | KEEP_PUNCT
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "validation", "test"):
        ds = load_dataset("google/WaxalNLP", args.config, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.filter(lambda s: s == args.speaker, input_columns="speaker_id")
        rows += [(split, r) for r in ds]
    print(f"{len(rows)} source clips for speaker {args.speaker}")

    # speaking rate from clips that need no cutting
    rates = []
    for _, r in rows:
        wav, sr = decode(r)
        sents = sentences(r["text"])
        if sents and len(wav) / sr <= args.max_sec:
            rates.append(len(" ".join(sents)) / (len(wav) / sr))
    rate = args.rate or float(np.median(rates))
    print(f"median speaking rate: {rate:.2f} chars/s (from {len(rates)} short clips)")

    drops, meta, hours = Counter(), [], Counter()
    for split, r in rows:
        wav, sr = decode(r)
        dur, sents = len(wav) / sr, sentences(r["text"])
        if not sents:
            drops["no text after normalization"] += 1
            continue
        if dur <= args.max_sec:
            parts, texts = split_by_pauses(wav, sr, 1), [" ".join(sents)]
        else:
            parts, texts = split_by_pauses(wav, sr, len(sents), args.min_pause), sents
            if parts is None:
                drops["long clip: too few pauses for its sentences"] += 1
                continue
            if not plausible(parts, texts, rate, min_sec=args.min_sec, max_sec=args.max_sec):
                drops["long clip: segments failed duration/rate check"] += 1
                continue
        if parts is None:
            drops["no speech found"] += 1
            continue
        for k, ((a, b), text) in enumerate(zip(parts, texts)):
            clean = to_model_text(text, allowed)
            if clean is None:
                drops["segment text has unsupported characters or digits"] += 1
                continue
            if not (args.min_sec <= b - a <= args.max_sec):
                drops["segment outside duration limits"] += 1
                continue
            seg = librosa.resample(wav[int(a * sr): int(b * sr)], orig_sr=sr, target_sr=TARGET_SR)
            peak = float(np.max(np.abs(seg))) or 1.0
            seg = seg * min(1.0, 0.95 / peak)
            name = f"{r['id']}_{k}.wav"
            sf.write(out / "wavs" / name, seg, TARGET_SR, subtype="PCM_16")
            meta.append((f"wavs/{name}", split, r["id"], clean, round(len(seg) / TARGET_SR, 3)))
            hours[split] += len(seg) / TARGET_SR / 3600

    with open(out / "metadata.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["file", "split", "source_id", "text", "seconds"])
        w.writerows(meta)
    report = {
        "speaker": args.speaker, "source_clips": len(rows), "median_rate_chars_per_s": round(rate, 3),
        "segments": Counter(m[1] for m in meta), "hours": {k: round(v, 3) for k, v in hours.items()},
        "dropped": dict(drops),
    }
    (out / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

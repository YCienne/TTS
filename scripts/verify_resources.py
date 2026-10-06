"""Verify the open items from docs/dataset_survey.md. Run on Colab (needs Hub access).

    python scripts/verify_resources.py --out reports/verify.json

Checks, in order:
  1. facebook/mms-tts-aka loads, and whether its vocabulary covers ɛ / ɔ.
  2. WAXAL twi_tts (and optionally fat_tts): hours, speakers, gender, sample rate,
     duration spread, character set, digits, and characters the MMS vocab lacks.
  3. Zero-shot synthesis with the unmodified MMS model (saved as a wav), which
     confirms the inference path and gives the "before fine-tuning" baseline.
The JSON report is what to paste back into the chat.
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
from akantts.normalize import charset, find_digits, normalize  # noqa: E402

MMS_ID = "facebook/mms-tts-aka"
WAXAL_ID = "google/WaxalNLP"
SMOKE_TEXTS = ["Ɛyɛ me deɛ sɛ mebɛkɔ fie.", "Akwaaba, ɛte sɛn?"]


def check_mms():
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained(MMS_ID)
    model = VitsModel.from_pretrained(MMS_ID)
    vocab = tok.get_vocab()
    return tok, model, {
        "vocab_size": len(vocab),
        "sampling_rate": model.config.sampling_rate,
        "has_epsilon": "ɛ" in vocab,
        "has_open_o": "ɔ" in vocab,
        "vocab_chars": sorted(vocab),
        "is_uroman": getattr(tok, "is_uroman", None),
    }


def check_split(config, split, vocab):
    from datasets import Audio, load_dataset

    ds = load_dataset(WAXAL_ID, config, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec dependency
    per_spk = defaultdict(lambda: {"n": 0, "sec": 0.0, "gender": set()})
    durs, rates, texts = [], Counter(), []
    for row in ds:
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        dur = len(wav) / sr
        durs.append(dur)
        rates[sr] += 1
        s = per_spk[row["speaker_id"]]
        s["n"] += 1
        s["sec"] += dur
        s["gender"].add(row.get("gender") or "")
        texts.append(row["text"])
    cs = charset(texts)
    d = np.array(durs)
    return {
        "utterances": len(ds),
        "hours": round(float(d.sum()) / 3600, 3),
        "sample_rates": dict(rates),
        "dur_sec": {k: round(float(v), 2) for k, v in
                    zip(("min", "p10", "median", "p90", "max"),
                        np.percentile(d, [0, 10, 50, 90, 100]))},
        "speakers": {k: {"n": v["n"], "hours": round(v["sec"] / 3600, 3),
                         "gender": sorted(v["gender"])}
                     for k, v in sorted(per_spk.items(), key=lambda kv: -kv[1]["sec"])},
        "n_with_digits": sum(find_digits(t) for t in texts),
        "charset": cs,
        "chars_missing_from_mms_vocab": {c: n for c, n in cs.items() if c not in vocab and c != " "},
        "sample_texts": texts[:5],
    }


def smoke_synthesis(tok, model, outdir):
    import torch

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, text in enumerate(SMOKE_TEXTS):
        inputs = tok(normalize(text), return_tensors="pt")
        with torch.no_grad():
            wav = model(**inputs).waveform[0].numpy()
        p = outdir / f"zeroshot_mms_aka_{i}.wav"
        sf.write(p, wav, model.config.sampling_rate)
        paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/verify.json")
    ap.add_argument("--configs", nargs="+", default=["twi_tts"],
                    help="WAXAL TTS configs, e.g. twi_tts fat_tts")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--audio-dir", default="reports/zeroshot")
    args = ap.parse_args()

    report = {}
    tok, model, report["mms_aka"] = check_mms()
    vocab = set(report["mms_aka"]["vocab_chars"])
    for cfg in args.configs:
        report[cfg] = {s: check_split(cfg, s, vocab) for s in args.splits}
    report["zeroshot_wavs"] = smoke_synthesis(tok, model, Path(args.audio_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

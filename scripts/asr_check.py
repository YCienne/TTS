"""Transcribe the prepared segments with MMS ASR (Akan adapter) and score them against their text.

    python scripts/asr_check.py --data /content/drive/MyDrive/akan_tts/data/speaker2

A segment cut in the wrong place pairs audio with text that is not what was said, so its
character error rate (CER) should be far higher than a correct one's. Writes asr_cer.tsv
(file, split, cer, reference, transcript) next to the metadata and prints the CER distribution.
ASR for Akan is imperfect, so even correct segments have a non-zero CER: look at the
distribution and listen to the worst segments before choosing a --drop-above threshold.
With --drop-above it also writes metadata_filtered.tsv.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.metrics import cer, wer  # noqa: E402

ASR_MODEL = "facebook/mms-1b-all"


def load_asr(device):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(ASR_MODEL, target_lang="aka")
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, target_lang="aka", ignore_mismatched_sizes=True)
    model = model.to(device).eval()

    def transcribe(wav, sr=16000):
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model(**inputs).logits.argmax(dim=-1)[0]
        return processor.decode(ids)

    return transcribe


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output folder of prepare_data.py")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--drop-above", type=float, default=None, help="CER above which segments are dropped")
    args = ap.parse_args()

    data = Path(args.data)
    with open(data / "metadata.tsv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["split"] in args.splits]
    transcribe = load_asr("cuda" if torch.cuda.is_available() else "cpu")

    scored = []
    for i, r in enumerate(rows):
        wav, sr = sf.read(data / r["file"], dtype="float32")
        hyp = transcribe(wav, sr)
        scored.append({**r, "cer": cer(r["text"], hyp), "wer": wer(r["text"], hyp), "transcript": hyp})
        if i % 100 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    with open(data / "asr_cer.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["file", "split", "cer", "wer", "reference", "transcript"])
        w.writerows([(s["file"], s["split"], round(s["cer"], 4), round(s["wer"], 4), s["text"], s["transcript"])
                     for s in scored])

    c = np.array([s["cer"] for s in scored])
    print(f"\n{len(c)} segments. CER percentiles (10/25/50/75/90/95): "
          f"{np.round(np.percentile(c, [10, 25, 50, 75, 90, 95]), 3).tolist()}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.8):
        print(f"  CER > {t}: {int((c > t).sum())} segments ({(c > t).mean():.0%})")

    if args.drop_above is not None:
        kept = [s for s in scored if s["cer"] <= args.drop_above]
        with open(data / "metadata_filtered.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(["file", "split", "source_id", "text", "seconds"])
            w.writerows([(s["file"], s["split"], s["source_id"], s["text"], s["seconds"]) for s in kept])
        print(f"kept {len(kept)}/{len(scored)} at CER <= {args.drop_above}")


if __name__ == "__main__":
    main()

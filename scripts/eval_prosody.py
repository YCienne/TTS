"""F0 (pitch) contour analysis: a prosody/naturalness proxy that CER/WER cannot provide.

    python scripts/eval_prosody.py --wavs outputs/eval_speaker3_kaggle --label speaker3 --out outputs/prosody
    python scripts/eval_prosody.py --model <checkpoint> --texts outputs/eval_speaker3_kaggle/eval_report.tsv \
        --label speaker2 --out outputs/prosody

Either point this at an existing directory of wavs (e.g. the ones eval_synthesis.py already wrote),
or give it --model and a prompt list and it will synthesize them itself -- unprocessed, the same way
eval_synthesis.py does, so the two speakers are compared on equal footing rather than one post-cleaned
and one not.

For each utterance this extracts the F0 (fundamental frequency / pitch) contour with librosa's pyin
and reports, over voiced frames only: mean F0, F0 standard deviation, and F0 range (max-min). Standard
deviation is the useful number here: flat, monotone, "robotic"-sounding speech has a near-zero F0
std-dev, while natural speech modulates pitch across a sentence. This is a proxy, not a naturalness
score -- it does not know whether the modulation lands on the right syllables, and this orthography
does not mark tone, so it cannot judge tonal *correctness* either. It only tells you whether the
voice is varying pitch at all, which a listener notices as "flat" or "expressive" without needing to
be a native speaker. Judge tone accuracy and genuine naturalness with human listeners instead.

Writes prosody_report.tsv (per utterance) and prints the aggregate mean F0 std-dev.
"""
import argparse
import csv
import sys
from pathlib import Path

REPORT_FIELDS = ["file", "voiced_frames", "f0_mean_hz", "f0_std_hz", "f0_range_hz"]


def _f0_stats(wav, sr):
    import librosa
    import numpy as np

    f0, voiced_flag, _ = librosa.pyin(wav, fmin=65, fmax=400, sr=sr)
    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced = voiced[~np.isnan(voiced)]
    if len(voiced) < 2:
        return None
    return {
        "voiced_frames": len(voiced),
        "f0_mean_hz": round(float(np.mean(voiced)), 1),
        "f0_std_hz": round(float(np.std(voiced)), 1),
        "f0_range_hz": round(float(np.max(voiced) - np.min(voiced)), 1),
    }


def _synthesize(model_path, texts, out_dir):
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from akantts.normalize import to_inference_text  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    vocab = set(tok.get_vocab()) | set(getattr(tok, "encoder", {}))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, raw in enumerate(texts):
        text, _ = to_inference_text(raw, vocab)
        if not text:
            continue
        torch.manual_seed(0)
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt").to(device)).waveform[0].cpu().numpy()
        path = out_dir / f"{i + 1:03d}.wav"
        sf.write(path, wav, model.config.sampling_rate)
        paths.append(path)
        print(f"synthesized {path}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", help="directory of existing wavs to analyze")
    ap.add_argument("--model", help="checkpoint to synthesize from instead of --wavs")
    ap.add_argument("--texts", help="eval_report.tsv (or any tsv with a 'text' column) giving prompts to synthesize")
    ap.add_argument("--label", required=True, help="speaker label for the printed summary")
    ap.add_argument("--out", default="outputs/prosody")
    args = ap.parse_args()

    out = Path(args.out) / args.label
    if args.wavs:
        wav_paths = sorted(Path(args.wavs).glob("*.wav"))
    elif args.model and args.texts:
        with open(args.texts, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        texts = [r["text"] for r in rows]
        wav_paths = _synthesize(args.model, texts, out)
    else:
        raise SystemExit("pass either --wavs <dir> or --model <checkpoint> with --texts <tsv>")

    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in wav_paths:
        wav, sr = sf.read(path)
        stats = _f0_stats(wav.astype("float32"), sr)
        if stats is None:
            print(f"{path.name}: too little voiced signal to track pitch, skipped")
            continue
        results.append({"file": path.name, **stats})
        print(f"{path.name}  f0 mean={stats['f0_mean_hz']:.1f}Hz std={stats['f0_std_hz']:.1f}Hz "
              f"range={stats['f0_range_hz']:.1f}Hz  ({stats['voiced_frames']} voiced frames)")

    report = out / "prosody_report.tsv"
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=REPORT_FIELDS, quoting=csv.QUOTE_NONE)
        w.writeheader()
        w.writerows(results)

    if results:
        mean_std = sum(r["f0_std_hz"] for r in results) / len(results)
        mean_mean = sum(r["f0_mean_hz"] for r in results) / len(results)
        print(f"\n{args.label}: {len(results)} utterances, mean F0 {mean_mean:.1f}Hz, "
              f"mean per-utterance F0 std-dev {mean_std:.1f}Hz (higher = more pitch variation)")
    print(f"report: {report}")


if __name__ == "__main__":
    main()

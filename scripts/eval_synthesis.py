"""Round-trip evaluation: synthesize held-out test sentences, transcribe them back with ASR,
and score CER/WER plus e<->ɛ and o<->ɔ vowel-harmony confusions.

    python scripts/eval_synthesis.py --model <checkpoint> --data <prepared data dir> --out outputs/eval

Prompts come from the dataset's own held-out **test** split (never seen in training, the same
split asr_check.py reports on), so there is no hand-picked wording to second-guess. Uses the
same ASR model (facebook/mms-1b-all, aka adapter) and the same akantts.metrics as asr_check.py.

Tone is not marked in this orthography, so neither the reference text nor the ASR transcript
carries it: this script cannot measure tonal correctness. Judge tone, and naturalness
generally, with the native-speaker listening test -- see docs/native_speaker_brief.md.

Writes eval_report.tsv (file, text, transcript, cer, wer, and the four confusion counts) and
prints the aggregate CER/WER and vowel-harmony confusion totals.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.metrics import cer, vowel_harmony_confusions, wer  # noqa: E402
from akantts.normalize import to_inference_text  # noqa: E402

ASR_MODEL = "facebook/mms-1b-all"
REPORT_FIELDS = ["file", "text", "transcript", "cer", "wer", "e_to_ɛ", "ɛ_to_e", "o_to_ɔ", "ɔ_to_o"]


def load_tts(model_path, device):
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    return tok, model


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


def load_prompts(data_dir, split, n):
    data = Path(data_dir)
    meta = data / "metadata_filtered.tsv" if (data / "metadata_filtered.tsv").exists() else data / "metadata.tsv"
    with open(meta, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["split"] == split]
    if not rows:
        raise SystemExit(f"no rows with split={split!r} in {meta}")
    return rows[:n], meta.name


def main():
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint folder or Hub id")
    ap.add_argument("--data", required=True, help="output folder of prepare_data.py")
    ap.add_argument("--split", default="test", help="which split to draw prompts from (default: held-out test)")
    ap.add_argument("--n", type=int, default=20, help="max number of prompts to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    rows, meta_name = load_prompts(args.data, args.split, args.n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | {len(rows)} prompts from {meta_name} (split={args.split})")

    tok, tts = load_tts(args.model, device)
    transcribe = load_asr(device)
    vocab = set(tok.get_vocab())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    totals = Counter()
    for i, r in enumerate(rows):
        text, removed = to_inference_text(r["text"], vocab)
        if removed:
            print(f"warning: {r['file']}: dropped characters not in the vocabulary: {removed}", file=sys.stderr)
        if not text:
            print(f"skip {r['file']}: nothing left to synthesize after cleanup")
            continue
        torch.manual_seed(args.seed)
        with torch.no_grad():
            wav = tts(**tok(text, return_tensors="pt").to(device)).waveform[0].cpu().numpy()
        sr = tts.config.sampling_rate
        path = out / f"{i + 1:03d}.wav"
        sf.write(path, wav, sr)
        hyp = transcribe(wav, sr)
        c, w = cer(text, hyp), wer(text, hyp)
        conf = vowel_harmony_confusions(text, hyp)
        totals.update(conf)
        results.append({"file": path.name, "text": text, "transcript": hyp, "cer": round(c, 4),
                         "wer": round(w, 4), **conf})
        print(f"{path.name}  cer={c:.3f} wer={w:.3f}  {text!r} -> {hyp!r}")

    report = out / "eval_report.tsv"
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=REPORT_FIELDS, quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(results)

    if results:
        mean_cer = sum(r["cer"] for r in results) / len(results)
        mean_wer = sum(r["wer"] for r in results) / len(results)
        print(f"\n{len(results)} prompts. mean CER {mean_cer:.3f}, mean WER {mean_wer:.3f}")
        print(f"vowel-harmony confusions (ASR transcript vs. the text asked to be said): {dict(totals)}")
    print("Tone is not marked in this orthography and is not covered here -- "
          "see docs/native_speaker_brief.md for the listening test.")
    print(f"report: {report}")


if __name__ == "__main__":
    main()

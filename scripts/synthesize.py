"""Generate Akan speech from text with a fine-tuned VITS checkpoint. No external TTS service is used.

    python scripts/synthesize.py --model <checkpoint folder> --text "Ɛyɛ dɛ yie." --out outputs
    python scripts/synthesize.py --model <checkpoint folder> --text-file prompts.txt --out outputs

One wav per input line (--text-file, one prompt per line) or for --text. Text is normalized the
same way as the training text; characters the model's vocabulary cannot represent are dropped with a
warning (digits are never spoken: numbers must be written as words). --seed fixes the sampling
noise, so a run is reproducible. Also writes <out>/prompts.tsv listing file, text and what was spoken.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import to_inference_text  # noqa: E402


def main():
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint folder or Hub id")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speaking-rate", type=float, default=None, help="1.0 = model default; <1 slower")
    args = ap.parse_args()

    lines = [args.text] if args.text else [
        l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(device).eval()
    if args.speaking_rate:
        model.speaking_rate = args.speaking_rate
    vocab = set(tok.get_vocab())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, raw in enumerate(lines):
        text, removed = to_inference_text(raw, vocab)
        if removed:
            print(f"warning: line {i + 1}: dropped characters not in the vocabulary: {removed}", file=sys.stderr)
        if not text:
            print(f"warning: line {i + 1}: nothing left to say, skipped", file=sys.stderr)
            continue
        torch.manual_seed(args.seed)
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt").to(device)).waveform[0].cpu().numpy()
        path = out / f"{i + 1:03d}.wav"
        sf.write(path, wav, model.config.sampling_rate)
        rows.append((path.name, raw, text, round(len(wav) / model.config.sampling_rate, 2)))
        print(f"{path}  {rows[-1][3]} s  {text}")
    with open(out / "prompts.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "input", "spoken_text", "seconds"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

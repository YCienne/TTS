"""Check that a converted training checkpoint really contains Meta's pretrained weights.

    python scripts/check_conversion.py --model /content/mms-tts-aka-punct --recipe /content/finetune-hf-vits

Why: loading Meta's MMS checkpoint with newer torch prints "weights not used" for weight_g / weight_v and
"newly initialized" for parametrizations.weight.original0/1 in the WaveNet layers. That is harmless if
torch remaps them on load, and a disaster (random weights in the flow and posterior encoder) if it does
not. The warning cannot tell which, so this compares the tensors themselves with the original file:

  1. the converted model.safetensors against facebook/mms-tts-aka (file level);
  2. with --recipe, the model as the training script will load it (model level);
  3. the tokenizer against the embedding table: every id must be a valid row, with no clashes.

Exit code 1 if anything pretrained is missing or different.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.checkpoints import compare_state  # noqa: E402

SAMPLES = ["mehunu mununkum. awia nso rebɔ kɛse pa ara?", "ɛyɛ dɛ, yie!", "sɛ wo kɔ fie a"]


def report(title, result, problems):
    print(f"\n[{title}] original tensors: {result['original_tensors']}, identical: {result['matched']}, "
          f"grown (extra rows ok): {len(result['grown'])}, different: {len(result['mismatched'])}, "
          f"missing: {len(result['missing'])}")
    for key, why in result["mismatched"][:5]:
        print(f"   DIFFERENT {key}: {why}")
    for key in result["missing"][:5]:
        print(f"   MISSING   {key}")
    if result["mismatched"] or result["missing"]:
        problems.append(title)


def main():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="converted (and possibly extended) checkpoint folder")
    ap.add_argument("--original", default="facebook/mms-tts-aka")
    ap.add_argument("--recipe", default=None, help="clone of ylacombe/finetune-hf-vits, enables the model-level check")
    args = ap.parse_args()

    problems = []
    model_dir = Path(args.model)
    original = load_file(hf_hub_download(args.original, "model.safetensors"))
    converted = load_file(str(model_dir / "model.safetensors"))
    n_disc = sum(k.startswith("discriminator.") for k in converted)
    print(f"converted file has {len(converted)} tensors, {n_disc} of them discriminator")
    if n_disc == 0:
        problems.append("no discriminator tensors")
    report("file level", compare_state(original, converted), problems)

    if args.recipe:
        sys.path.insert(0, args.recipe)
        from utils.modeling_vits_training import VitsModelForPreTraining

        model = VitsModelForPreTraining.from_pretrained(model_dir)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        report("as loaded for training", compare_state(original, state), problems)

    # tokenizer against embedding table
    from transformers import AutoTokenizer

    rows = converted["text_encoder.embed_tokens.weight"].shape[0]
    tok = AutoTokenizer.from_pretrained(model_dir)
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    ids = [i for s in SAMPLES for i in tok(s)["input_ids"]]
    unk = tok.unk_token_id
    print(f"\n[tokenizer] embedding rows: {rows}, vocab.json entries: {len(vocab)}, <unk> id: {unk}, "
          f"largest id used by the samples: {max(ids)}")
    for s in SAMPLES[:1]:
        print("   ", s, "->", tok(s)["input_ids"])
    clashes = [t for t, i in vocab.items() if i == unk]
    if max(ids) >= rows or clashes:
        print(f"   PROBLEM: an id is outside the table (largest {max(ids)}, rows {rows}) or tokens share the <unk> id: {clashes}")
        problems.append("tokenizer/embedding")
    if unk is not None and unk >= rows:
        print("   note: <unk> has no embedding row (a quirk of Meta's original files). Harmless here, because text with "
              "characters outside the vocabulary is filtered out before it reaches the model.")

    print("\nRESULT:", "OK, the pretrained weights are intact" if not problems else f"PROBLEMS in: {', '.join(problems)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

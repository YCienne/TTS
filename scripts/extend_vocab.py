"""Add the tokens ". , ? !" to a converted MMS training checkpoint.

    python scripts/extend_vocab.py --src /content/mms-tts-aka-train --dst /content/mms-tts-aka-punct \
        --recipe /content/finetune-hf-vits

The MMS Akan vocabulary has 31 tokens and no punctuation, so punctuation would become <unk>.
This grows the text encoder's embedding table by one row per new token (the old rows are kept;
the new ones are initialised like VITS initialises them), bumps config.vocab_size and appends the
tokens to vocab.json. The recipe's own model class is used so the discriminator survives.
Untested against the real checkpoint when written: run it, then check the printed sizes.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

NEW_TOKENS = [".", ",", "?", "!"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="converted checkpoint folder")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--recipe", required=True, help="clone of ylacombe/finetune-hf-vits")
    args = ap.parse_args()

    sys.path.insert(0, args.recipe)
    import torch
    from torch import nn
    from utils.modeling_vits_training import VitsModelForPreTraining

    src, dst = Path(args.src), Path(args.dst)
    vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    new = [t for t in NEW_TOKENS if t not in vocab]

    model = VitsModelForPreTraining.from_pretrained(src)
    old = model.text_encoder.embed_tokens
    n_old = old.num_embeddings
    assert max(vocab.values()) == n_old - 1, f"vocab ids (max {max(vocab.values())}) do not match embedding rows ({n_old})"
    emb = nn.Embedding(n_old + len(new), old.embedding_dim, padding_idx=old.padding_idx)
    with torch.no_grad():
        emb.weight[:n_old] = old.weight
        nn.init.normal_(emb.weight[n_old:], mean=0.0, std=old.embedding_dim ** -0.5)
    model.text_encoder.embed_tokens = emb
    model.config.vocab_size = n_old + len(new)
    vocab.update({t: n_old + i for i, t in enumerate(new)})

    model.save_pretrained(dst)
    for f in src.iterdir():  # tokenizer and feature-extractor files
        if not (dst / f.name).exists() and f.is_file():
            shutil.copy(f, dst / f.name)
    (dst / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"embedding rows {n_old} -> {n_old + len(new)}; added {new}; saved to {dst}")


if __name__ == "__main__":
    main()

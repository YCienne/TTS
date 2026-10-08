"""Add the tokens ". , ? !" to a converted MMS training checkpoint.

    python scripts/extend_vocab.py --src /content/mms-tts-aka-train --dst /content/mms-tts-aka-punct \
        --recipe /content/finetune-hf-vits

The MMS Akan vocabulary has 30 entries and no punctuation, so punctuation would become <unk>. The
tokenizer puts <unk> at id 30, one past the last embedding row. This grows the text encoder's embedding
table by one row for <unk> plus one per new token (old rows are kept; new ones are initialised like VITS
initialises them), bumps config.vocab_size and appends the tokens to vocab.json after <unk>'s id, so no two
tokens share an id (akantts.checkpoints.plan_new_tokens). The recipe's own model class is used so the
discriminator survives. Run scripts/check_conversion.py afterwards.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.checkpoints import plan_new_tokens  # noqa: E402

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
    added_file = src / "added_tokens.json"
    added = json.loads(added_file.read_text(encoding="utf-8")) if added_file.exists() else {}

    model = VitsModelForPreTraining.from_pretrained(src)
    old = model.text_encoder.embed_tokens
    n_old = old.num_embeddings
    assert max(vocab.values()) == n_old - 1, f"vocab ids (max {max(vocab.values())}) do not match embedding rows ({n_old})"
    new_vocab, extra = plan_new_tokens(vocab, added, NEW_TOKENS, n_old)
    emb = nn.Embedding(n_old + extra, old.embedding_dim, padding_idx=old.padding_idx)
    with torch.no_grad():
        emb.weight[:n_old] = old.weight
        nn.init.normal_(emb.weight[n_old:], mean=0.0, std=old.embedding_dim ** -0.5)
    model.text_encoder.embed_tokens = emb
    model.config.vocab_size = n_old + extra
    new = [t for t in NEW_TOKENS if t not in vocab]
    vocab = new_vocab

    model.save_pretrained(dst)
    for f in src.iterdir():  # tokenizer and feature-extractor files
        if not (dst / f.name).exists() and f.is_file():
            shutil.copy(f, dst / f.name)
    (dst / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"embedding rows {n_old} -> {n_old + extra}; added {new} at ids {[vocab[t] for t in new]}; "
          f"<unk> id {added.get('<unk>')}; saved to {dst}")


if __name__ == "__main__":
    main()

"""Checks and planning for MMS/VITS checkpoints, independent of torch so they can be unit-tested."""
import numpy as np


def canon_key(key: str) -> str:
    """Name weight-norm tensors the legacy way (weight_g, weight_v).

    Newer torch stores them as ``parametrizations.weight.original0/1``; the same tensors appear
    under either name depending on which torch API saved the file.
    """
    return key.replace("parametrizations.weight.original0", "weight_g").replace(
        "parametrizations.weight.original1", "weight_v")


def compare_state(original: dict, converted: dict, atol: float = 1e-6, skip_prefixes=("discriminator.",)) -> dict:
    """Compare every tensor of ``original`` with the tensor of the same canonical name in ``converted``.

    ``converted`` may have extra tensors (the discriminator) and extra leading rows in a tensor
    (a grown embedding table); the shared rows must match. Returns counts and the problems found.
    """
    conv = {canon_key(k): v for k, v in converted.items() if not k.startswith(tuple(skip_prefixes))}
    matched, mismatched, missing, grown = 0, [], [], []
    for key, ref in original.items():
        name = canon_key(key)
        if name not in conv:
            missing.append(key)
            continue
        a, b = np.asarray(ref, dtype=np.float64), np.asarray(conv[name], dtype=np.float64)
        if a.shape != b.shape:
            if a.ndim == b.ndim and a.shape[1:] == b.shape[1:] and b.shape[0] > a.shape[0]:
                grown.append(key)
                b = b[: a.shape[0]]
            else:
                mismatched.append((key, f"shape {a.shape} vs {b.shape}"))
                continue
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        if diff > atol:
            mismatched.append((key, f"max abs diff {diff:.3g}"))
        else:
            matched += 1
    return {"original_tensors": len(original), "matched": matched, "mismatched": mismatched,
            "missing": missing, "grown": grown}


def plan_new_tokens(vocab: dict, added: dict, new_tokens, n_rows: int):
    """Plan token ids and extra embedding rows for ``new_tokens``.

    Returns ``(new_vocab, extra_rows)``. The original MMS tokenizer puts ``<unk>`` at id ``n_rows``,
    one past the last embedding row, so an unknown character would index out of range. When that is
    the case, the first extra row is reserved for ``<unk>`` and the new tokens come after it, so no
    two tokens share an id.
    """
    extra = 1 if added.get("<unk>") == n_rows else 0
    out = dict(vocab)
    for tok in new_tokens:
        if tok not in vocab:
            out[tok] = n_rows + extra
            extra += 1
    return out, extra

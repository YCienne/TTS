import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.checkpoints import canon_key, compare_state, plan_new_tokens  # noqa: E402

RNG = np.random.default_rng(0)


def tensors():
    return {
        "flow.flows.0.wavenet.in_layers.0.weight_g": RNG.standard_normal((4, 1, 1)),
        "flow.flows.0.wavenet.in_layers.0.weight_v": RNG.standard_normal((4, 2, 5)),
        "text_encoder.embed_tokens.weight": RNG.standard_normal((30, 8)),
        "decoder.conv_pre.bias": RNG.standard_normal(6),
    }


def test_canon_key_unifies_both_weight_norm_namings():
    assert canon_key("a.parametrizations.weight.original0") == "a.weight_g"
    assert canon_key("a.parametrizations.weight.original1") == "a.weight_v"
    assert canon_key("a.weight_g") == "a.weight_g" and canon_key("decoder.conv_pre.bias") == "decoder.conv_pre.bias"


def test_identical_weights_match_across_naming_and_ignore_the_discriminator():
    orig = tensors()
    conv = {canon_key(k).replace("weight_g", "parametrizations.weight.original0")
            .replace("weight_v", "parametrizations.weight.original1"): v for k, v in orig.items()}
    conv["discriminator.layer.weight"] = RNG.standard_normal(3)
    r = compare_state(orig, conv)
    assert r["matched"] == 4 and not r["mismatched"] and not r["missing"]


def test_randomly_initialised_weights_are_caught():
    orig = tensors()
    conv = {k: v.copy() for k, v in orig.items()}  # an exact copy, then one tensor replaced by fresh random values
    conv["flow.flows.0.wavenet.in_layers.0.weight_g"] = RNG.standard_normal((4, 1, 1))
    r = compare_state(orig, conv)
    assert r["matched"] == 3 and len(r["mismatched"]) == 1 and "weight_g" in r["mismatched"][0][0]


def test_missing_tensor_is_reported():
    orig = tensors()
    r = compare_state(orig, {k: v for k, v in orig.items() if k != "decoder.conv_pre.bias"})
    assert r["missing"] == ["decoder.conv_pre.bias"]


def test_grown_embedding_keeps_old_rows_and_flags_changed_ones():
    orig = tensors()
    grown = np.vstack([orig["text_encoder.embed_tokens.weight"], RNG.standard_normal((5, 8))])
    r = compare_state(orig, {**orig, "text_encoder.embed_tokens.weight": grown})
    assert r["grown"] == ["text_encoder.embed_tokens.weight"] and not r["mismatched"]
    changed = grown.copy()
    changed[3] += 1.0  # an old row was altered
    assert compare_state(orig, {**orig, "text_encoder.embed_tokens.weight": changed})["mismatched"]


def test_plan_new_tokens_reserves_the_unk_row_and_avoids_id_clashes():
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz1234")}  # 30 ids: 0..29
    new_vocab, extra = plan_new_tokens(vocab, {"<unk>": 30}, [".", ",", "?", "!"], 30)
    assert extra == 5  # <unk> keeps id 30 and gets a row; four new tokens follow
    assert [new_vocab[t] for t in ".,?!"] == [31, 32, 33, 34]
    assert len(set(new_vocab.values())) == len(new_vocab) and 30 not in new_vocab.values()


def test_plan_new_tokens_without_unk_clash_and_with_existing_tokens():
    vocab = {"a": 0, "b": 1, ".": 2}
    new_vocab, extra = plan_new_tokens(vocab, {}, [".", ","], 3)
    assert extra == 1 and new_vocab == {"a": 0, "b": 1, ".": 2, ",": 3}

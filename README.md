# Akan (Twi) text-to-speech

Fine-tunes a pretrained VITS model (`facebook/mms-tts-aka`) on single-speaker Twi speech from WAXAL (`twi_tts`).
Built for the University of Ghana HCI Lab TTS task; deadline Sun 18 Oct 2026, 11:59 pm GMT.

**Status:** two fine-tuning runs have completed on different WAXAL speakers with an identical recipe — 100
training epochs each, confirmed from the Hub-hosted `hparams.yml` training logs for both runs — on speaker 2
and on speaker 3 (`speaker3_punct`, more and cleaner usable audio — see `docs/dataset_survey.md`). **Speaker 3
is the final submitted checkpoint**, chosen after listening to both: it has less audible high-frequency buzz
and far fewer vowel-harmony confusions (7 vs. 23 of 20 prompts), though speaker 2 has a slightly lower
character error rate (CER 0.211 vs. 0.235) and drops fewer whole words on some sentences — automated
round-trip evaluation (`scripts/eval_synthesis.py`, CER/WER and ɛ/e, ɔ/o vowel-harmony confusions on the
held-out test split) found real tradeoffs between the two rather than one speaker winning outright; see
`docs/report.docx` section 3 for the full comparison. A separate F0 (pitch) contour check
(`scripts/eval_prosody.py`, same 20 prompts, raw model output) found comparable pitch variation on both
speakers (mean per-utterance F0 std-dev 52.3 Hz for speaker 2, 50.4 Hz for speaker 3) — neither sounds
flatter than the other on this proxy, so it is not what's driving the speaker-3 preference below; the large
gap in mean F0 (240.5 Hz vs. 177.7 Hz) just reflects the two speakers' different voice registers. The
structured, written native-speaker listening
protocol (naturalness, intelligibility, tone and vowel-harmony judgments this orthography and ASR cannot
measure — see `docs/native_speaker_brief.md`) was distributed to 3 raters using speaker-2 samples, prepared
before this speaker-3 decision, so its scored results (once returned) will speak most directly to speaker 2
rather than the final submitted checkpoint; results pending regardless. Separately, we also ran an
informal, unscored in-person listening session with the same 3 native speakers: all three identified the
output as Asante Twi, and all three found speaker 3 more audible, with clearer pronunciation and tone, and
understood more of what was said, than with speaker 2 — though still with room for improvement. That's
qualitative support for the speaker-3 and Asante-Twi calls above, not a substitute for the scored protocol.

Both checkpoints are published on Hugging Face for provenance (speaker 3, final: `Lhue-Lhue/akan-tts-speaker3-vits`;
speaker 2: `Lhue-Lhue/akan-tts-mms-vits`), and the Gradio demo (`scripts/demo_app.py`, `space/app.py`)
speaks with speaker 3. Synthesized audio is run through a light cleanup pass (`src/akantts/postprocess.py`:
spectral-gate denoise, a gentle high-frequency cut, peak normalization) to reduce the buzz/ringing this scale
of fine-tuning leaves in. That measurably improves how clean the output sounds but is not a substitute for
more training data — a checkpoint fine-tuned on well under an hour of audio will not reach the naturalness of
a production TTS system trained on tens of hours; there is no standard percentage score for "how human"
synthesized speech sounds (the field uses MOS, a 1–5 listening-test score, not a percentage), so this README
does not claim one. Separately, words or letter-clusters that are rare or absent in the ~1h training corpus
mispronounce regardless of cleanup -- e.g. "Akwaaba" appears exactly once across the full 872-utterance WAXAL
`twi_tts` training split, spoken by a different (unused) speaker, so neither fine-tuned checkpoint ever learned
it; the demo examples below avoid it for this reason.

## Layout
```
docs/dataset_survey.md      dataset, licence and checkpoint survey
scripts/verify_resources.py, audit_data.py   dataset and speaker checks (Colab)
scripts/prepare_data.py     cut long clips into sentence segments, clean text, resample
scripts/asr_check.py        score every segment with MMS ASR to catch wrong cuts
scripts/make_audiofolder.py, extend_vocab.py   dataset and vocabulary for the training recipe
scripts/synthesize.py       inference: text in, wav out (no external TTS)
scripts/eval_synthesis.py   round-trip ASR eval on held-out test prompts: CER/WER and ɛ/e, ɔ/o vowel-harmony confusions
scripts/eval_prosody.py     F0 (pitch) contour check: mean/std-dev per utterance, a naturalness/monotony proxy
scripts/demo_app.py         Gradio demo: type Akan text, hear the fine-tuned model speak it
notebooks/00..05            Colab runners, in order (verify, audit, prepare, training prereqs, ASR check, fine-tune)
src/akantts/                normalize, segment, audio_quality, postprocess, metrics
configs/                    data selection and training config
tests/                      unit tests (pytest)
```
Evaluation is automated CER/WER plus a vowel-harmony-specific confusion count (`eval_synthesis.py`) and an
F0 pitch-variance proxy (`eval_prosody.py`), on the held-out test split, together with a native-speaker
listening test for naturalness and tone, which this orthography does not mark and the automated checks
cannot see (`docs/native_speaker_brief.md`).
Hugging Face checkpoint upload and the written report (`docs/report.docx`) are done; the presentation deck is
still planned.

## Setup
```bash
pip install -r requirements.txt
pytest -q
```
Data download and preparation (notebooks 00-04) run on Colab, because the data hosts need Hub access.
Fine-tuning itself ran on Colab for speaker 2 and on Kaggle for speaker 3, the final checkpoint
(`scripts/colab_run_all.py` / `scripts/kaggle_run_all.py`); both single NVIDIA T4 GPU -- Kaggle's default
dual-T4 runtime hit a multi-GPU NCCL hang partway through a run, so the Kaggle script forces single-GPU
(`CUDA_VISIBLE_DEVICES=0`) to match the Colab-validated recipe.

## Acknowledgements and licences
- **Base model:** [`facebook/mms-tts-aka`](https://huggingface.co/facebook/mms-tts-aka), Meta MMS (Pratap et al., 2023), CC-BY-NC-4.0. Non-commercial use only.
- **Data:** [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) (Google Research with the University of Ghana), `twi_tts`, CC-BY-4.0.
- **Training code:** the VITS training loop, discriminator code and checkpoint conversion are from [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) (MIT licence, copyright 2023 Yoach Lacombe), cloned at run time and not copied here. `scripts/patch_recipe.py` applies two one-line compatibility fixes to that clone (a single-speaker `KeyError` and a newer-matplotlib `AttributeError`, both found by running the smoke test); see `docs/dataset_survey.md` for details.
- **Speech recogniser used for checks:** `facebook/mms-1b-all` with the `aka` adapter (CC-BY-NC-4.0).
- **Libraries:** Hugging Face `transformers`, `datasets`, `accelerate`.
- Our own work: text normalization, segmentation of long clips, data selection and filtering, ASR-based alignment check, vocabulary extension, training configuration, inference script, evaluation, and this repository.

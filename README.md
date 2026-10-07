# Akan (Twi) text-to-speech

Fine-tunes a pretrained VITS model (`facebook/mms-tts-aka`) on single-speaker Twi speech from WAXAL (`twi_tts`).
Built for the University of Ghana HCI Lab TTS task; deadline Sun 18 Oct 2026, 11:59 pm GMT.

**Status:** data prepared (speaker 2, about 2 h); fine-tuning code written but not yet run. Nothing is trained yet. See `docs/dataset_survey.md` for the dataset and checkpoint decisions.

## Layout
```
docs/dataset_survey.md      dataset, licence and checkpoint survey
scripts/verify_resources.py, audit_data.py   dataset and speaker checks (Colab)
scripts/prepare_data.py     cut long clips into sentence segments, clean text, resample
scripts/asr_check.py        score every segment with MMS ASR to catch wrong cuts
scripts/make_audiofolder.py, extend_vocab.py   dataset and vocabulary for the training recipe
scripts/synthesize.py       inference: text in, wav out (no external TTS)
notebooks/00..05            Colab runners, in order (verify, audit, prepare, training prereqs, ASR check, fine-tune)
src/akantts/                normalize, segment, audio_quality, metrics
configs/                    data selection and training config
tests/                      unit tests (pytest)
```
Planned: evaluation of synthesized speech, demo app, report.

## Setup
```bash
pip install -r requirements.txt
pytest -q
```
Training and data download run on Colab, because the data hosts need Hub access.

## Acknowledgements and licences
- **Base model:** [`facebook/mms-tts-aka`](https://huggingface.co/facebook/mms-tts-aka), Meta MMS (Pratap et al., 2023), CC-BY-NC-4.0. Non-commercial use only.
- **Data:** [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) (Google Research with the University of Ghana), `twi_tts`, CC-BY-4.0.
- **Training code:** the VITS training loop, discriminator code and checkpoint conversion are from [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) (MIT licence, copyright 2023 Yoach Lacombe), cloned at run time and not copied here.
- **Speech recogniser used for checks:** `facebook/mms-1b-all` with the `aka` adapter (CC-BY-NC-4.0).
- **Libraries:** Hugging Face `transformers`, `datasets`, `accelerate`.
- Our own work: text normalization, segmentation of long clips, data selection and filtering, ASR-based alignment check, vocabulary extension, training configuration, inference script, evaluation, and this repository.

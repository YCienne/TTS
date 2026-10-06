# Akan (Twi) text-to-speech

Fine-tunes a pretrained VITS model (`facebook/mms-tts-aka`) on single-speaker Twi speech from WAXAL (`twi_tts`).
Built for the University of Ghana HCI Lab TTS task; deadline Sun 18 Oct 2026, 11:59 pm GMT.

**Status:** planning and scaffolding. Nothing is trained yet. See `docs/dataset_survey.md` for the dataset and checkpoint decisions.

## Layout
```
docs/dataset_survey.md      dataset, licence and checkpoint survey
scripts/verify_resources.py checks the survey's open items (run on Colab)
notebooks/00_verify_resources.ipynb   Colab runner for the above
src/akantts/normalize.py    text normalization (NFC, ɛ/ɔ canonicalisation)
configs/data.yaml           data selection
tests/                      unit tests (pytest)
```
Planned: data preparation, fine-tuning, inference/demo, and evaluation modules.

## Setup
```bash
pip install -r requirements.txt
pytest -q
```
Training and data download run on Colab, because the data hosts need Hub access.

## Acknowledgements and licences
- **Base model:** [`facebook/mms-tts-aka`](https://huggingface.co/facebook/mms-tts-aka), Meta MMS (Pratap et al., 2023), CC-BY-NC-4.0. Non-commercial use only.
- **Data:** [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) (Google Research with the University of Ghana), `twi_tts`, CC-BY-4.0.
- **Libraries:** Hugging Face `transformers` and `datasets`. The fine-tuning approach follows the `ylacombe/finetune-hf-vits` recipe.
- Our own work: normalization, data selection and filtering, training configuration, evaluation, and this repository.

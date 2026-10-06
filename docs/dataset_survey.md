# Akan TTS: dataset and checkpoint survey

Last updated 2026-10-06. Items marked **[confirmed]** were read from the Hugging Face Hub (dataset card, config table,
model metadata). Items marked **[VERIFY]** are still unconfirmed and are checked by `scripts/verify_resources.py` on Colab,
because the sandbox can't download audio or run the models.

## Candidate datasets

| Dataset | Variety | Quality | Licence | Status |
|---|---|---|---|---|
| **WAXAL `twi_tts`**, `google/WaxalNLP` | Twi **[confirmed]** (Asante vs Akuapem: **[VERIFY]**, ask raters) | Studio-like, voice actors reading a phonetically balanced script **[confirmed]** | CC-BY-4.0 (provider: University of Ghana) **[confirmed]** | train 872 / val 117 / test 104 utterances, ~511 MB train parquet **[confirmed]**. **[confirmed on Colab]** 8.86 h train (+1.07 h val, 1.00 h test), 4 speakers (1 M, 2 F, 3 M, 4 F; 2.0–2.6 h each in train), 48 kHz. See "Colab verification" below. |
| **WAXAL `fat_tts`** | Fante **[confirmed]** | Same pipeline | CC-BY-4.0 | train 953 / val 117 / test 101, ~516 MB train. A second Akan variety, useful for a cross-variety test. |
| **BibleTTS**, OpenSLR 129 | Asante Twi, Akuapem Twi | 48 kHz studio, single speaker, verse-aligned | CC-BY-SA | Not checked (openslr.org is blocked). Large (tens of hours). Religious register, and share-alike applies to derivatives. Kept as a fallback or ablation. |
| WAXAL `aka_asr` / `aka_asr_v2` | Akan | Natural speech, many speakers | CC-BY-4.0 | **Not for TTS training.** Useful as held-out data for the ASR round-trip evaluation. Test split is speaker-disjoint **[confirmed]**. |

Note: there is **no `aka_tts` config**. The Akan TTS data is the separate `twi_tts` and `fat_tts` configs.

## Pretrained models

- **`facebook/mms-tts-aka`** exists **[confirmed]**: VITS, 36.3M parameters, `transformers` (`VitsModel`), **CC-BY-NC-4.0**. This is our fine-tuning start point. The licence is non-commercial, which is fine for this assessment and must be stated in the report and the README.
- `facebook/mms-tts-twi` does **not** exist **[confirmed]**.
- The MMS Akan vocabulary **does** contain ɛ and ɔ **[confirmed on Colab]**. It has 31 tokens, is lowercase, runs at 16 kHz, and is not uroman-based.
- Existing Twi TTS models on the Hub (GhanaNLP `nano-twi`, `stable-twi-tts`, Kasanoma) can be cited as prior work. They must not be used to generate our outputs.

## Recommendation (updated)

1. **Data:** WAXAL `twi_tts`, single speaker. Pick the speaker with the most clean hours, then filter by duration and SNR. With only ~1.1k utterances in total, expect 1–4 h per speaker, which is enough for VITS **fine-tuning** but not for training from scratch.
2. **Model:** fine-tune `facebook/mms-tts-aka` (VITS) with the Hugging Face `transformers` VITS training recipe (adapted from `ylacombe/finetune-hf-vits`).
3. **Zero-shot baseline:** run the unmodified MMS aka model on the held-out sentences. This gives a clean before/after comparison for the report.
4. **Evaluation audio:** WAXAL `aka_asr` test (speaker-disjoint) for ASR-based intelligibility checks, plus our own unseen sentences.

## Colab verification (2026-10-06)

| Finding | Consequence |
|---|---|
| MMS vocab has ɛ and ɔ but no `. , ? ! : ;`, digits other than 2 and 3, or the letters c j v z q x | Extend the embedding table with `. , ? !` so prosody cues survive; deal with the rare letters (below). |
| Clips are long: median 16.5 s, p90 ~99 s, max 380 s. Mean is 36.6 s. | VITS trains on clips of roughly 20 s or less. Most of the 8.9 h sits in long clips, so usable hours per speaker are well below 2 h. Measured by `scripts/audit_data.py`. |
| 43% of train utterances contain digits, written as bracketed annotations such as `[10] Du [×] ahodoɔ [10] du [=] ma yɛn [100] ɔha.` | The number words are written beside the brackets, so `normalize()` drops `[...]`. On the samples seen, no digits remain. The audit confirms this across all utterances. |
| `ↄ` (U+2184) appears 91 times across train/val/test | It is a mis-encoded ɔ. Mapped to ɔ in `normalize()`. |
| Orthography uses final ɛ/ɔ (`deɛ`, `ahodoɔ`, `Ghanafoɔ`) | Consistent with the Asante Twi convention, but **not confirmed**. A native speaker should check the audio and spelling. |
| Same 4 speakers in train, validation and test | Fine for TTS. It means validation is not a held-out-speaker test. |

## Colab audit (2026-10-06)

Train split, per speaker. "Usable" is hours in clips of at most 20 s. The SNR-like figure is the gap between the loudest and quietest 10% of frames, so treat it as a ranking, not a measurement.

| Speaker | Gender | Hours | Usable <=20 s | Usable <=30 s | SNR-like (dB) | Noise floor (dB) | Silence | chars/s |
|---|---|---|---|---|---|---|---|---|
| 1 | M | 2.13 | 0.36 (17%) | 0.52 | 40.3 | -61.6 | 47% | 8.1 |
| **2** | F | 2.62 | 0.38 (14%) | 0.59 | **50.7** | **-78.4** | 44% | 7.6 |
| 3 | M | 2.08 | 0.58 (28%) | 0.66 | 43.6 | -65.3 | 42% | 7.8 |
| 4 | F | 2.03 | 0.41 (20%) | 0.69 | 35.1 | -56.5 | 29% | 10.3 |

- No clipping anywhere. Speaker 2 is the cleanest, speaker 4 the noisiest and fastest.
- **Usable data is the problem.** At most 0.58 h of any speaker is in clips of 20 s or less. Training on those alone would be about 20–35 minutes of one voice. Hence `scripts/prepare_data.py`, which cuts long clips at sentence pauses.
- After normalization only 17 of 872 train utterances still contain digits (bracket stripping works). Left over: `c` (90 utterances), `j` (51), `"` (49), `v` (41), `;` (31), `z` (25), brackets and a few others. `;` and `:` become commas, quotes and brackets are dropped, and utterances with `c j v z q x` or bare digits are rejected.
- **Pitch is not usable yet.** Median F0 looked plausible for speaker 2 (219 Hz) and 1 (139 Hz), but within-clip spread of 9–12 semitones (normal read speech is about 2–4) means the tracker made octave errors. The "possible mixed voices" flag fired on all four speakers, so it was removed. Pitch should not drive the speaker choice.

## Constraints

- The sandbox can't reach the data hosts or run models, so data prep and training run on **Colab free tier**.
- 3 native-speaker raters: report per-rater MOS and make no significance claims.
- The 18 Oct 2026 deadline leaves about 12 days from 6 Oct.

## Sources

- WAXAL dataset card: https://huggingface.co/datasets/google/WaxalNLP · paper https://arxiv.org/abs/2602.02734
- UGSpeechData (source of the Ghana data): https://doi.org/10.57760/sciencedb.22298
- MMS Akan: https://huggingface.co/facebook/mms-tts-aka · MMS paper https://arxiv.org/abs/2305.13516
- BibleTTS: https://openslr.org/129 · https://arxiv.org/abs/2207.03546

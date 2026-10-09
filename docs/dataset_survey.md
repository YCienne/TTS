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

## Data preparation, speaker 2 (2026-10-06)

**Run 1 (cut at the N-1 longest pauses): failed to recover long clips.** 256 source clips gave 143 train + 16 validation + 14 test segments, 0.45 h in total, about what the clips under 20 s alone provide. 100 long clips failed the duration/rate check and only 1 had too few pauses, so the pauses exist but the wrong ones were chosen. Likely cause: the speaker also pauses inside sentences, and those breath pauses can be longer than the gaps between sentences. 14 segments were dropped for unsupported characters or digits.

**Fix:** `choose_cuts` picks the pauses that make each segment's duration proportional to its sentence's length in characters (dynamic programming, with a small preference for longer pauses). On 150 synthetic clips with breath pauses of 0.25–0.9 s inside sentences, every cut was right in 83% of clips, against 16% for the longest-pauses rule. The old rule's synthetic yield is close to the ~15% yield seen on real data, which supports the diagnosis. The 83% is on synthetic audio, so real yield is still unmeasured. `--longest-pauses` reproduces the old behaviour for comparison, and `long_clip_diagnostics.tsv` records the outcome per long clip.

**Run 2 (length-aware cuts): yield doubled.** 364 train + 25 validation + 32 test segments = 0.97 h (train 0.84 h). Of the 117 long clips, 58 passed, 33 failed "speaking rate too high", 25 failed "segment too long", 1 had too few pauses. 21 segments were dropped for unsupported characters or digits.

Two problems in the diagnostics. (1) The rate reference was biased: the 6.79 chars/s median came from whole short clips with leading and trailing silence, while trimmed segments run at 9–11 chars/s, so the 1.8x limit rejected clips that were probably cut correctly. The reference is now measured on trimmed speech spans. (2) Some failures are single sentences longer than 20 s (for example one 27.5 s clip with 1 sentence); these are now retried by also cutting at commas (`--no-clauses` turns that off). Both changes are untested on the real data. Whether a cut is *correct*, as opposed to passing the checks, is only known by listening.

**Run 3 (rate calibrated on trimmed spans, comma fallback): 2.02 h.** 907 train + 101 validation + 71 test segments = 1.70 h train, 0.19 h validation, 0.14 h test, from 256 source clips. Median speaking rate on trimmed spans is 8.66 chars/s. Of the 117 long clips, 85 passed at sentence level and 14 more at comma level, 16 still had a segment over 20 s, 1 had a segment under 1 s and 1 had too few pauses. 31 segments were dropped for unsupported characters or digits. The comma-level cuts are the least certain: one recovered clip produced segments of 1.1 to 1.4 s.

Passing the duration and rate checks does not prove the cuts are right. `scripts/asr_check.py` (notebook 04) transcribes every segment with MMS ASR (`facebook/mms-1b-all` has an `aka` adapter) and scores it against its text; wrongly cut segments should show a much higher error rate. The same code gives the ASR round-trip intelligibility metric for synthesized speech later.

**ASR check of the cuts (speaker 2, 1079 segments, MMS ASR with the `aka` adapter).** CER percentiles 10/25/50/75/90/95 were 0.04 / 0.07 / **0.10** / 0.14 / 0.22 / 0.30. The distribution is unimodal with a thin tail and no second hump near 0.8 to 1.0, which is what mostly-correct cuts look like. Segments above CER 0.3, 0.4, 0.5, 0.6 and 0.8: 55 (5%), 31 (3%), 24 (2%), 20 (2%), 7 (1%). Residual ASR error on correct audio is unknown, and a boundary that is off by one word probably scores about 0.2 to 0.3 on a 60-character segment, so a threshold of 0.3 is a deliberate choice of quality over quantity: it drops about 5% of segments (the tail also contains correct audio the recogniser mishears, such as names and loanwords). Not yet checked: whether the worst segments are wrong cuts or ASR mistakes. The reference and ASR text of the worst six answer that.

**Worst segments, read as text.** The six worst (CER 0.85 to 2.2) are real misalignments, not just recogniser errors. In two clips the ASR transcript of a segment equals the *reference text of its neighbour*: `twi_272_1` holds the speech of sentence 0 and `twi_503_21` holds the speech of sentence 22, while the segments that should hold those sentences (`twi_272_0`, `twi_503_22`) are noise (ASR gives strings such as "popoaonoaina"). So the failure mode is a cut shifted by one sentence, mostly at the start or end of long clips (`twi_503` has 23 segments). A fragment such as `twi_301_2` ("nanso,", 6 characters) scores badly because one wrong letter is a large share of a short reference, so CER is unreliable on very short segments.

**Applied threshold: CER <= 0.3.** Kept 1024 of 1079 segments: train 855 of 907 (1.63 h of 1.70 h), validation 99 of 101 (0.18 h), test 70 of 71 (0.14 h). Only about 4% of the hours are lost. Not done: dropping the neighbours of a flagged segment, or the whole clip. A shifted cut should push its neighbours' scores up too, so they would be caught by the same threshold; if listening to the fine-tuned model reveals clipped words, tighten this first.

**Training recipe.** No ready-made Akan training checkpoint exists on the Hub (`ylacombe/mms-tts-*-train` covers eng, fra, kor, mar, guj, spa, tam, acd and a few others), but one can be built: the recipe's `convert_original_discriminator_checkpoint.py --language_code aka` downloads the generator and the discriminator from `facebook/mms-tts/full_models/aka/` (`D_100000.pth`, 561 MB, present on the Hub). The older `aka.tar.gz` download has only the generator, which is why the first check looked negative. Colab (T4, 15 GB) ships torch 2.11 and transformers 5.18, while the recipe asks for `transformers>=4.35.1`, was written in early 2024 and imports private transformers internals, so notebook 05 builds a pinned virtual environment (torch 2.4.1, transformers 4.44.2, datasets 2.21.0) instead of trusting the new versions. The `ylacombe/finetune-hf-vits` recipe reports good fine-tunes from 80–150 samples, so about an hour of data is workable. HF `transformers` `VitsModel` has no training loss, so the training loop and discriminator come from that open-source repo, which the report must acknowledge as reused code.

**If the yield is still low:** forced alignment with a CTC model (torchaudio `MMS_FA`) would remove the dependence on pauses. It needs a GPU runtime, and it can't be tested in the sandbox.

## Colab runtime (checked 2026-10-08)

A connection-test report from the GPU runtime: Tesla T4, 15360 MiB, driver 580.82.07; system Python 3.13.15 (so the training recipe runs in a uv-built Python 3.11 environment, because torch 2.4.1 has no wheels for 3.13); 12 GiB RAM; 71 GiB free disk. Drive was not mounted in that cell, so the Drive listing was empty; this does not show whether the prepared data survived. Earlier runtimes (CPU-only, `cuda False`) could not train.

**Dataset loader bug (datasets 2.21.0 with pandas 3).** Loading the audiofolder failed with "`file_name` key must be a string". Reproduced in a throwaway environment: the loader reads CSV metadata through pandas, pandas 3 yields Arrow large_string, and the loader rejects it; the same dataset as `metadata.jsonl` loads fine. `make_audiofolder.py` now writes JSON Lines, and the training environment pins `pandas<3`.

**First conversion on Colab (2026-10-08).** The converted training checkpoint has 46.7M parameters (generator plus discriminator) and a 30-row embedding table. Two findings. (1) The MMS vocabulary has 30 entries, not 31: `<unk>` is a tokenizer-level token with id 30, one past the last embedding row (Meta's quirk; unknown characters would index out of range, which our text filtering prevents). The first extension gave `.` the same id 30 as `<unk>`; it now reserves its own row for `<unk>` and puts `. , ? !` at ids 31 to 34 (embedding rows 30 -> 35). (2) Loading Meta's file with newer torch printed "weights not used" for `weight_g`/`weight_v` and "newly initialized" for `parametrizations.weight.original0/1` in the WaveNet layers of the flow and posterior encoder. That is harmless if torch remaps them and a silent loss of the pretrained weights if not; `scripts/check_conversion.py` compares the actual tensors with Meta's original, at file level and as the training script loads the model, and must print `RESULT: OK` before training.

**Conversion check result (2026-10-08).** On the converted checkpoint, `check_conversion.py` found all 762 of Meta's generator tensors identical, both in the saved file and as the training script loads them (0 different, 0 missing, 74 discriminator tensors present, 1 embedding table grown). The "weights not used / newly initialized" warnings were therefore cosmetic: torch remapped `weight_g`/`weight_v` on load. The same run also flagged the old extension layout (34 rows, `.` sharing id 30 with `<unk>`), which is what the check is for; the fixed layout (35 rows, `. , ? !` at ids 31 to 34) needs the extended checkpoint to be rebuilt under the new cache name `mms-tts-aka-punct2`.

**Rebuilt `punct2` checkpoint, verified on the real T4 runtime (2026-10-09).** Re-ran the fixed `05_finetune.ipynb` (pulled fresh from GitHub, since Colab does not update a notebook already open from a stale Drive copy) on a new T4 runtime. `extend_vocab.py` produced the expected layout: `embedding rows 30 -> 35; added ['.', ',', '?', '!'] at ids [31, 32, 33, 34]`, no clash with `<unk>`. `check_conversion.py` then printed `RESULT: OK, the pretrained weights are intact` (762/762 identical, both at file level and as loaded for training). The dataset built cleanly from the filtered speaker-2 metadata: 855 / 99 / 70 train/validation/test rows.

**Two bugs in the external training recipe, found by actually running the 2-epoch smoke test end to end, both fixed by `scripts/patch_recipe.py` (applied in notebook cell 2, right after cloning, and in `colab_run_all.py`; never by editing our own data or config):**

1. `KeyError: 'speaker_id'` at the first training step. The recipe's data collator sets `batch["speaker_id"] = None` when the dataset has no `speaker_id` column, which is our case: `facebook/mms-tts-aka` is a true single-speaker checkpoint (`num_speakers: 1`, `speaker_embedding_size: 0`), and we never pass `speaker_id_column_name`. But three places in `run_vits_finetuning.py` (the forward call in the training loop, and two more in generation/sampling code) read `batch["speaker_id"]` with plain dict indexing rather than `.get()`, and by the time the batch reaches those points the key is gone, so Python raises `KeyError` on the very first optimisation step — this would hit any single-speaker fine-tune on this recipe, not something specific to our data. Patched all three sites to `batch.get("speaker_id")`, which is exactly the `None` the collator already intended and exactly what a single-speaker model's forward pass expects.
2. `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` during the end-of-training alignment plot — after all 108 optimisation steps and validation finish, but before the checkpoint saves, so a run could reach 100% and still lose its checkpoint to this crash. `FigureCanvasAgg.tostring_rgb()` was removed from newer matplotlib (the pinned version here postdates the recipe). Patched both occurrences in `utils/plot.py` to the modern `buffer_rgba()`, slicing off the alpha channel to keep the existing 3-channel (H, W, 3) contract the rest of the code expects.

**2-epoch smoke test result (2026-10-09).** With both patches applied, training ran all 108 steps (2 epochs x 855 examples, batch 16) to completion in ~2m20s on the T4, followed by validation on both eval batches and a final checkpoint save: `model.safetensors` (332 MB) plus config/tokenizer files written to `Drive/akan_tts/runs/speaker2_punct/`. `scripts/synthesize.py` then loaded that checkpoint and produced a 4.34 s sample for "mehunu mununkum. awia nso rebɔ kɛse pa ara." — confirming the full pipeline (environment, recipe, conversion, vocabulary extension, conversion check, dataset build, training, inference) runs end to end on this setup. Next: the real 100-epoch run (`EPOCHS = 100` in the notebook).

## Constraints

- The sandbox can't reach the data hosts or run models, so data prep and training run on **Colab free tier**.
- 3 native-speaker raters: report per-rater MOS and make no significance claims.
- The 18 Oct 2026 deadline leaves about 12 days from 6 Oct.

## Sources

- WAXAL dataset card: https://huggingface.co/datasets/google/WaxalNLP · paper https://arxiv.org/abs/2602.02734
- UGSpeechData (source of the Ghana data): https://doi.org/10.57760/sciencedb.22298
- MMS Akan: https://huggingface.co/facebook/mms-tts-aka · MMS paper https://arxiv.org/abs/2305.13516
- BibleTTS: https://openslr.org/129 · https://arxiv.org/abs/2207.03546

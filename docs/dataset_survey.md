# Akan TTS: dataset and checkpoint survey

Status: **desk research via web search, 2026-10-06.** The sandbox blocks huggingface.co, openslr.org and arxiv.org,
so dataset cards were *not* read directly. Everything marked **VERIFY** must be checked on Colab before we rely on it.

## Candidate datasets

| Dataset | Variety | Quality | Licence | Notes |
|---|---|---|---|---|
| **WAXAL TTS**, `google/WaxalNLP` (Univ. of Ghana is the Akan provider) | Akan (Twi; Fante is listed separately) | Studio-like, single-speaker voice actors, phonetically balanced script | CC-BY-4.0 | Up to ~16 h per voice actor. **VERIFY:** the Akan config name, hours, speakers, sample rate (24 kHz?), dialect. |
| **BibleTTS**, OpenSLR 129 (Meyer et al., Interspeech 2022) | Asante Twi and Akuapem Twi | 48 kHz studio, single speaker, verse-aligned | CC-BY-SA | Up to ~86 h per language. **VERIFY:** Twi hours and speaker gender. The reading style is religious and archaic, and CC-BY-SA applies share-alike to derivatives. |
| `ghananlpcommunity/twi-speech-text-multispeaker-16k` | Twi | 16 kHz, multispeaker, forced-aligned | **VERIFY** | ~21k pairs. Probably better for ASR evaluation than TTS training. |
| `ghananlpcommunity/akuapem_multispeaker_audio_transcribed` | Akuapem | Multispeaker | **VERIFY** | Backup. |

## Pretrained models and prior work

- **MMS-TTS** (`facebook/mms-tts`, VITS per language): a search result says Akan (`aka`) is supported, but I could not confirm that `facebook/mms-tts-aka` exists. **VERIFY first**: this decides the fine-tune starting point. The licence is CC-BY-NC 4.0, which is fine for coursework but should be stated.
- **Existing Twi TTS** (GhanaNLP `nano-twi`, `stable-twi-tts`, Kasanoma): useful as comparison baselines and for citation. Do **not** use them to generate our outputs, since the brief forbids external TTS and we must train ourselves.
- **Fallback bases if no `aka` checkpoint exists:** `mms-tts-eng` or another MMS voice with the Akan alphabet added (`ylacombe/finetune-hf-vits` recipe), or Piper `libritts_r` warm-start.

## Recommendation

1. **Primary:** WAXAL Akan TTS, single best voice, 3–5 h subset. It is CC-BY, studio quality and made for TTS, and it matches current speakers better than Bible reading.
2. **Alternative or ablation:** BibleTTS Asante Twi subset, to compare against and to cover the case where WAXAL turns out to be Akuapem-heavy or thin on hours.
3. Report the variety **as verified from the data**, not assumed.
4. Choose the speaker by SNR, pitch stability and transcript quality, then document the choice.

## Constraints discovered

- **The sandbox cannot reach the data hosts.** Data prep and training must run on **Colab (free tier)**; the repo only holds scripts. Free T4 has roughly 12 h sessions and may disconnect, so use short epochs, frequent checkpoints to Drive, and a 2–4 h data subset.
- **Listener study:** 3 native speakers, so report MOS with per-rater scores and no significance claims.

## Next actions

1. A Colab cell to verify the open items: `facebook/mms-tts-aka` exists, the WAXAL Akan config, hours and speakers, and BibleTTS Twi sizes.
2. Scaffold the repo (`configs/`, `src/`, `notebooks/`) and a smoke-test inference notebook.
3. Write the normalization module (NFC, ɛ/ɔ canonicalization) with unit tests.

## Sources

- WAXAL: https://huggingface.co/datasets/google/WaxalNLP · https://arxiv.org/abs/2602.02734
- BibleTTS: https://openslr.org/129 · https://arxiv.org/abs/2207.03546
- MMS-TTS: https://huggingface.co/facebook/mms-tts
- GhanaNLP: https://huggingface.co/ghananlpcommunity

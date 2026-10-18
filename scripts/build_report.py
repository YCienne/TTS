"""Build the submission report (docs/report.docx) from python-docx.

Structure follows the task brief's required sections exactly: Data and preprocessing, Modeling
methodology, Evaluation and results, Discussion and limitations (<= 4 pages), then References on
a new page (excluded from that limit, per the brief).

The structured, scored native-speaker listening test (docs/native_speaker_brief.md) was distributed
but had not returned results by submission -- see section 3.2 and 4, which discuss that gap
directly rather than leaving a placeholder. Separately, an informal in-person listening session with
the same three native speakers was conducted and is reported in 3.2 as qualitative, unscored
evidence; it is not a substitute for the structured protocol. If rater responses to the structured
protocol land before the deadline, update sections 3.2 and 4 with their actual findings and re-run
this script.

    python scripts/build_report.py
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

def add_heading(doc, text, level=1):
    return doc.add_heading(text, level=level)


def add_bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def build():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(5)
    bullet_style = doc.styles["List Bullet"]
    bullet_style.font.size = Pt(10.5)
    bullet_style.paragraph_format.space_after = Pt(3)
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    title = doc.add_heading("Akan (Twi) Text-to-Speech: Fine-Tuning Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("University of Ghana HCI Lab TTS task")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author = doc.add_paragraph("Lucienne Awoudi")
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph().add_run(
        "Akan variety: Asante Twi, identified from the orthography of the WAXAL twi_tts transcripts -- "
        "final ɛ and ɔ spellings (e.g. deɛ, ahodoɔ) are consistent with the Asante Twi convention, "
        "and three native speakers independently confirmed this by ear in an informal listening "
        "session (3.2). See the Discussion section."
    ).italic = True

    # --- Data and preprocessing -------------------------------------------------
    add_heading(doc, "1. Data and Preprocessing", level=1)
    doc.add_paragraph(
        "Dataset: WAXAL [3], google/WaxalNLP, config twi_tts (Google Research with the University "
        "of Ghana, CC-BY-4.0). Studio-recorded, single speakers reading a phonetically balanced "
        "script. We fine-tuned on two of the four available speakers with an identical recipe: "
        "speaker 2 (female), the cleanest by a loud/quiet-frame SNR-like proxy (50.7 dB), and "
        "speaker 3 (male), who has the most usable short-clip audio of the four after filtering to "
        "clips of 20 seconds or less (0.58 h, 28% of his 2.1 h, versus 0.38 h / 14% for speaker 2) "
        "despite a lower SNR-like figure (43.6 dB) and is the speaker whose checkpoint we ultimately "
        "submit as final (section 3 explains why). Most of the raw 8.9 hours sits in clips much "
        "longer than 20 s (VITS trains on short clips), so both runs relied on "
        "scripts/prepare_data.py's sentence-boundary cutting to recover usable audio from those "
        "longer recordings; see docs/dataset_survey.md for the full per-speaker table."
    )
    doc.add_paragraph(
        "Dataset justification: WAXAL is the only openly licensed, studio-quality Akan TTS corpus "
        "we found with per-utterance text alignment and a permissive licence (CC-BY-4.0) compatible "
        "with redistributing a derivative model. Its train/validation/test split (872/117/104 "
        "utterances) let us hold out genuine unseen text for every evaluation in this report, "
        "rather than hand-writing test sentences."
    )
    doc.add_paragraph("Preprocessing steps (scripts/prepare_data.py, src/akantts/):", style="Heading 2")
    add_bullets(doc, [
        "Resampled 48 kHz source audio to 16 kHz (the rate facebook/mms-tts-aka was trained at).",
        "Split long recordings into sentence-length clips at pause boundaries.",
        "Text normalization: NFC Unicode, canonical ɛ/ɔ spelling (mapping look-alike "
        "Greek and mis-encoded characters such as ↄ onto the correct Akan letters), removal "
        "of bracketed digit annotations (the spoken number words sit beside the brackets, so the "
        "brackets themselves are never read aloud), lowercasing to match the MMS vocabulary.",
        "No phonemization: the MMS Akan checkpoint is a grapheme-based (not phoneme-based) model, "
        "so we normalize spelling rather than convert to a phoneme set.",
        "Vocabulary extension: the base MMS vocabulary has no punctuation tokens; we added "
        ". , ? ! so that sentence-level prosody cues survive fine-tuning (scripts/extend_vocab.py).",
        "An ASR-based alignment check (scripts/asr_check.py, facebook/mms-1b-all) flagged "
        "segments whose audio did not match their transcript, to catch bad cuts before training.",
    ])

    # --- Modeling methodology ----------------------------------------------------
    add_heading(doc, "2. Modeling Methodology", level=1)
    doc.add_paragraph(
        "Architecture: VITS [2] (Conditional Variational Autoencoder with Adversarial Learning), "
        "an end-to-end model that jointly learns text-to-spectrogram prediction and the vocoder: a "
        "HiFi-GAN-style stack of transposed convolutions decodes the flow-based acoustic "
        "representation directly into a waveform, so there is no separate vocoder stage to choose "
        "or train. We fine-tuned the pretrained checkpoint facebook/mms-tts-aka [1] (Meta MMS, "
        "36.3M parameters, CC-BY-NC-4.0) rather than training from scratch, given our limited (~2 "
        "hour) single-speaker dataset."
    )
    doc.add_paragraph(
        "Reused vs. our own work: the VITS training loop, discriminator code, and checkpoint "
        "conversion come from the open-source recipe ylacombe/finetune-hf-vits [4] (MIT licence), "
        "cloned at run time and not copied into our repository; we applied two one-line "
        "compatibility fixes to that clone (a single-speaker KeyError and a newer-matplotlib "
        "AttributeError), both found by running its own smoke test. Our own work is the data "
        "selection and filtering, text normalization, segmentation, vocabulary extension, training "
        "configuration, the inference script, and the full evaluation harness (automated and "
        "native-speaker)."
    )
    doc.add_paragraph("Key hyperparameters (configs/train_aka.json):", style="Heading 2")
    add_bullets(doc, [
        "100 training epochs, batch size 16 (train and eval), gradient accumulation 1.",
        "Learning rate 2e-5, AdamW with beta1=0.8, beta2=0.99, warmup ratio 0.01.",
        "VITS loss weights: discriminator 3, feature-maps 1, generator 1, KL 1.5, duration 1, "
        "mel-spectrogram 35.",
        "Mixed precision (fp16), seed 456, clips capped at 1-20 s and 500 tokens.",
    ])
    doc.add_paragraph(
        "Computing environment: speaker 2 trained on Google Colab (single NVIDIA T4 GPU), writing "
        "checkpoints directly to Google Drive so a disconnect does not lose progress and converted "
        "checkpoints are cached there too, avoiding a re-download of the base model after a runtime "
        "reset. Speaker 3 -- the final submitted checkpoint -- trained on Kaggle instead "
        "(scripts/kaggle_run_all.py), also on a single NVIDIA T4: Kaggle's default runtime offers "
        "two T4 GPUs, but a dual-GPU attempt hung partway through (an NCCL ALLGATHER deadlock, a "
        "multi-GPU interconnect issue rather than a code bug), so the script forces single-GPU "
        "execution (CUDA_VISIBLE_DEVICES=0) to match the recipe as validated on Colab. Kaggle has "
        "no Drive mount, so outputs are instead written to the kernel's own working directory and "
        "pulled down after the run completes."
    )

    # --- Evaluation and results ---------------------------------------------------
    add_heading(doc, "3. Evaluation and Results", level=1)
    doc.add_paragraph("3.1 Automated (round-trip ASR)", style="Heading 2")
    doc.add_paragraph(
        "scripts/eval_synthesis.py synthesizes held-out test-split prompts, transcribes them back "
        "with facebook/mms-1b-all [1] (aka adapter), and scores character/word error rate (CER/"
        "WER) plus a vowel-harmony-specific confusion count (e↔ɛ, o↔ɔ). This "
        "gives an automated, repeatable measure of intelligibility and pronunciation, but it is an "
        "upper bound on true audio quality (the ASR model itself is imperfect for Akan), and it "
        "cannot measure naturalness or tone at all, since this orthography does not mark tone and "
        "the ASR transcript inherits that gap."
    )
    doc.add_paragraph(
        "Speaker 2 (comparison run, published but not demo-selectable), 20 held-out test-split "
        "prompts (scripts/eval_synthesis.py, seed 0): mean CER 0.211, mean WER 0.654. "
        "Vowel-harmony confusions in the ASR transcript, counted against the text the model was "
        "asked to say: e→ɛ 4, ɛ→e 6, o→ɔ 7, ɔ→o 6 (23 confusions total)."
    )
    doc.add_paragraph(
        "Speaker 3 (final submitted checkpoint), same 20 held-out test-split prompts and "
        "procedure: mean CER 0.235, mean WER 0.617. Vowel-harmony confusions: e→ɛ 2, "
        "ɛ→e 0, o→ɔ 2, ɔ→o 3 (7 total). Speaker 2's CER is slightly lower "
        "and it drops fewer whole words on some sentences, which CER/WER do not isolate from "
        "substitution errors -- on that axis speaker 2 is the stronger checkpoint. We nonetheless "
        "submit speaker 3 as final: its vowel-harmony confusions are far fewer (7 vs. 23) and, as "
        "3.2 below notes, it has less audible high-frequency buzz, and we weighted that clarity and "
        "pronunciation accuracy over speaker 2's edge in word-completeness. This is a genuine "
        "tradeoff rather than a one-sided win; section 4 discusses it further. WER being "
        "substantially higher than CER for both speakers is expected for a tonal, "
        "agglutinative-leaning orthography scored word-by-word -- a single wrong vowel can flip a "
        "whole word while leaving most characters correct. As noted above, facebook/mms-1b-all is "
        "itself an imperfect ASR model for Akan, so these numbers are an upper bound on the audio's "
        "true error, useful for ranking and spotting outliers rather than as an absolute score. "
        "Full per-prompt results for speaker 3 are in outputs/eval_speaker3_kaggle/eval_report.tsv."
    )
    doc.add_paragraph(
        "Beyond the automated numbers, informal listening across both checkpoints found the same "
        "pattern: speaker 3 sounds less synthetic overall -- less audible high-frequency "
        "buzz/ringing -- but speaker 2 more reliably pronounces every syllable of a word rather "
        "than clipping or slurring it. We weighted speaker 3's clarity and pronunciation accuracy "
        "more heavily for the final submission (see above). The interactive demo therefore speaks "
        "with speaker 3; speaker 2 remains published on Hugging Face for comparison but is not "
        "demo-selectable. All synthesized output (both the demo and scripts/synthesize.py) is run "
        "through a light cleanup pass (src/akantts/postprocess.py: a spectral-gate denoise, a "
        "gentle high-frequency cut, and peak normalization) to reduce buzz on either checkpoint's "
        "output; this measurably cleans up the signal but does not change pronunciation, prosody, "
        "or the underlying data ceiling -- there is no standard percentage score for \"how human\" "
        "synthesized speech is "
        "(the field uses MOS, a 1-5 listening-test rating, not a percentage), and we do not claim "
        "one here."
    )
    doc.add_paragraph(
        "As a further automated proxy -- not a tone or naturalness score -- we extracted each "
        "clip's F0 (pitch) contour with librosa's pYIN tracker (scripts/eval_prosody.py) on the "
        "same 20 prompts and the models' raw output. Speaker 2: mean F0 240.5 Hz, std-dev 52.3 Hz. "
        "Speaker 3: mean F0 177.7 Hz, std-dev 50.4 Hz. Pitch variation is comparable between the "
        "two (52.3 vs. 50.4 Hz) -- neither sounds flatter -- so this does not explain the clarity "
        "and tone preference for speaker 3 in 3.2; the mean-F0 gap instead reflects the speakers' "
        "different voice registers. Like CER/WER, this shows whether pitch varies, not whether it "
        "is tonally correct, since this orthography does not mark tone; see 3.2 for human judgment."
    )

    doc.add_paragraph("3.2 Native-speaker listening test", style="Heading 2")
    doc.add_paragraph(
        "Tone and perceived naturalness are outside what the automated metric in 3.1 can cover -- "
        "this orthography does not mark tone, and ASR-based scoring cannot judge how natural a clip "
        "sounds. To cover both, we designed a three-task listening protocol "
        "(docs/native_speaker_brief.md) and distributed it to three native speakers. Task 1 "
        "sanity-checks the natural recordings and a zero-shot (untrained) baseline, and identifies "
        "the speaker's variety. Task 2 blind-rates 24 anonymized clips (8 natural, 8 zero-shot "
        "baseline, 8 fine-tuned, from the same 8 held-out sentences) on naturalness and "
        "intelligibility (1-5 each) and logs mispronounced words, with particular attention to tone "
        "and vowel harmony. Task 3 elicits new unseen test sentences in the natural speaker's "
        "variety."
    )
    doc.add_paragraph(
        "Raters had not returned their written responses to this structured protocol by the "
        "submission deadline, so this report cannot present their scored naturalness, "
        "intelligibility, or tone judgments here; Section 4 discusses this gap directly. For the "
        "record, the Task 2 clips are drawn from the speaker-2 checkpoint, since the brief and "
        "samples were prepared before the speaker-3 decision below -- once returned, these results "
        "will speak most directly to the fine-tuning recipe in general and to speaker 2 "
        "specifically."
    ).italic = True
    doc.add_paragraph(
        "Separately, we ran an informal, unscored in-person session with the same three "
        "listeners: all three identified the output as Asante Twi (corroborating page 1), and all "
        "three found speaker 3 more audible, with clearer pronunciation and tone, and understood "
        "more of what was said, than speaker 2 -- while still noting room for improvement. This is "
        "qualitative feedback from three listeners in an unscored, non-blind setting, not the "
        "blind Task 2 ratings, so we treat it as supporting evidence rather than a substitute for "
        "the structured protocol's pending results."
    )

    # --- Discussion and limitations -----------------------------------------------
    add_heading(doc, "4. Discussion and Limitations", level=1)
    doc.add_paragraph(
        "Failure modes visible in the automated evaluation (3.1): vowel-harmony confusions are "
        "asymmetric and the pattern differs by speaker -- speaker 2's confusions split fairly evenly "
        "across all four directions (e→ɛ 4, ɛ→e 6, o→ɔ 7, ɔ→o 6), while speaker 3's smaller set "
        "skews toward closing the vowel (e→ɛ 2, o→ɔ 2, ɔ→o 3, no ɛ→e at all). We read this as "
        "the model defaulting to whichever realization was more frequent in its training data for "
        "each case, rather than a systematic harmony-direction bias -- with only 20 prompts and an "
        "imperfect ASR evaluator, this is a pattern worth confirming on more data, not a settled "
        "conclusion. WER running well above CER reflects how one wrong vowel can flip a whole word "
        "while leaving most characters correct -- a word-level failure CER understates. The audible "
        "high-frequency buzz (3.1) is a vocoder-training artifact, not a pronunciation error -- "
        "present throughout a clip regardless of content, consistent with a GAN-based vocoder not "
        "fully converged on this little data; cleanup (3.1) reduces "
        "but does not eliminate it. We cannot report scored tone-specific failures, since tone is not "
        "marked in the orthography and the structured listening test (3.2) had not returned "
        "numeric results by submission -- still the most significant gap here. The informal "
        "session (3.2) gave a matching qualitative signal -- clearer tone for speaker 3 -- but "
        "three unscored listeners cannot replace that protocol, so any stronger claim would be "
        "speculation."
    )
    doc.add_paragraph(
        "A concrete word-level failure traced to root cause: \"Akwaaba\" (a common greeting) is "
        "consistently clipped or incomplete on both checkpoints. It appears exactly once in the "
        "full 872-utterance WAXAL twi_tts training split -- mid-sentence in a folk-tale spoken by "
        "speaker 4, whom we did not fine-tune on -- so neither checkpoint's training data contains "
        "it at all; the model is extrapolating an unfamiliar letter sequence rather than "
        "reproducing a learned pattern. This is a data-coverage gap, not a training or architecture "
        "deficiency, and more epochs or a different speaker would not fix it; we removed the word "
        "from the demo examples for this reason and expect the same for any other word this rare."
    )
    doc.add_paragraph("Known limitations:", style="Heading 2")
    add_bullets(doc, [
        "Tone is not marked in this orthography and so is not modeled explicitly; only human "
        "listening -- not automated metrics -- can evaluate it. Three listeners informally judged "
        "speaker 3's tone clearer than speaker 2's (3.2), but the scored protocol has not yet "
        "returned results.",
        "ɛ and ɔ pronunciation depends on correct spelling surviving normalization; normalize() "
        "maps look-alike/mis-encoded characters (Greek ε/ς, ↄ) to the correct letters, but other "
        "encodings need the same treatment.",
        "Single speaker only, no voice-cloning support, and well under an hour of usable training "
        "audio for the final checkpoint -- small for TTS fine-tuning.",
        "Digits are not expanded to words; numbers must already be spelled out.",
        "The base model's licence (CC-BY-NC-4.0) restricts the checkpoint to non-commercial use.",
        "The Akan variety (Asante Twi, page 1) rests on our own orthographic determination, "
        "informally corroborated by three native speakers by ear (3.2); structured Task 1 "
        "verification had not returned a result by submission.",
        "Audio naturalness is bounded by how little training data this fine-tuning used; cleanup "
        "(3.1) improves perceived clarity but cannot substitute for more data or longer training.",
        "Rare or unseen letter sequences mispronounce regardless of checkpoint choice (see the "
        "\"Akwaaba\" example above) -- a corpus-coverage limit, not a quality dial we can turn.",
    ])
    doc.add_paragraph("Suggested improvements:", style="Heading 2")
    add_bullets(doc, [
        "Complete the structured, scored native-speaker listening protocol "
        "(docs/native_speaker_brief.md) for tone, naturalness, and variety judgments beyond the "
        "informal session already run (3.2).",
        "More training audio, including rare greetings, and a second speaker to test "
        "generalization.",
        "An explicit tone-marking/prediction step evaluated against native-speaker judgments, "
        "plus digit-expansion normalization.",
    ])

    doc.add_page_break()
    add_heading(doc, "References", level=1)
    doc.add_paragraph(
        "(Excluded from the four-page limit, per the submission guidelines.)"
    ).italic = True
    add_bullets(doc, [
        "[1] V. Pratap, A. Tjandra, B. Shi, et al., \"Scaling Speech Technology to 1,000+ "
        "Languages,\" arXiv:2305.13516, 2023. (Massively Multilingual Speech -- the base TTS "
        "checkpoint facebook/mms-tts-aka and the ASR model facebook/mms-1b-all used for "
        "evaluation.)",
        "[2] J. Kim, J. Kong, and J. Son, \"Conditional Variational Autoencoder with Adversarial "
        "Learning for End-to-End Text-to-Speech,\" arXiv:2106.06103, 2021. (The VITS architecture "
        "underlying the base model.)",
        "[3] A. Diack, P. Nelson, K. Agbesi, et al., \"WAXAL: A Large-Scale Multilingual African "
        "Language Speech Corpus,\" arXiv:2602.02734, 2026. (Source of the twi_tts training and "
        "evaluation data.)",
        "[4] Y. Lacombe, finetune-hf-vits (GitHub repository), 2023. "
        "https://github.com/ylacombe/finetune-hf-vits (Training recipe we fine-tuned with, MIT "
        "licence.)",
        "[5] T. Wolf, L. Debut, V. Sanh, et al., \"HuggingFace's Transformers: State-of-the-Art "
        "Natural Language Processing,\" arXiv:1910.03771, 2019. (transformers library used "
        "throughout for model loading, tokenization and inference.)",
    ])

    doc.add_paragraph()
    add_heading(doc, "Deliverables", level=2)
    add_bullets(doc, [
        "Code, data preparation, evaluation harness and documentation: "
        "https://github.com/YCienne/TTS",
        "Fine-tuned checkpoint (speaker 3, final submitted model): "
        "https://huggingface.co/Lhue-Lhue/akan-tts-speaker3-vits",
        "Speaker-2 checkpoint, reported above for comparison, not demo-selectable: "
        "https://huggingface.co/Lhue-Lhue/akan-tts-mms-vits",
        "Interactive demo (Gradio): scripts/demo_app.py, space/app.py -- speaks with the speaker-3 "
        "checkpoint only",
    ])
    doc.add_paragraph(
        "A shared folder collecting audio samples, text prompts, and the model checkpoint together "
        "(per the submission guidelines) will accompany this report with the final submission form; "
        "in the interim, the checkpoint and generated samples are reachable from the links above."
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "report.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    build()

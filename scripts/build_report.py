"""Build the submission report (docs/report.docx) from python-docx.

Structure follows the task brief's required sections exactly: Data and preprocessing, Modeling
methodology, Evaluation and results, Discussion and limitations (<= 4 pages), then References on
a new page (excluded from that limit, per the brief).

Fills in everything we know for certain and leaves clearly marked placeholders for numbers that
depend on outputs this script does not have access to (eval_report.tsv's CER/WER, and the
native-speaker ratings) -- see the PENDING markers below. Re-run after those land.

    python scripts/build_report.py
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

PENDING_COLOR = RGBColor(0xB0, 0x00, 0x00)


def add_heading(doc, text, level=1):
    return doc.add_heading(text, level=level)


def add_pending(doc, text):
    p = doc.add_paragraph()
    run = p.add_run("[PENDING -- " + text + "]")
    run.italic = True
    run.font.color.rgb = PENDING_COLOR
    return p


def add_bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def build():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(10.5)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    title = doc.add_heading("Akan (Twi) Text-to-Speech: Fine-Tuning Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("University of Ghana HCI Lab TTS task")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_pending(doc, "author name(s)")

    doc.add_paragraph(
        "Akan variety: "
    ).add_run(
        "provisionally Asante Twi -- the orthography (final ɛ/ɔ in words such as "
        "deɛ, ahodoɔ) is consistent with the Asante Twi convention, but this has not "
        "been confirmed by a native speaker."
    ).italic = True
    add_pending(doc, "confirm the variety with a native-speaker rater (brief Task 1, question 1) "
                      "and replace the provisional statement above")

    # --- Data and preprocessing -------------------------------------------------
    add_heading(doc, "1. Data and Preprocessing", level=1)
    doc.add_paragraph(
        "Dataset: WAXAL [3], google/WaxalNLP, config twi_tts (Google Research with the University "
        "of Ghana, CC-BY-4.0). Studio-recorded, single speakers reading a phonetically balanced "
        "script. We fine-tuned on two of the four available speakers with an identical recipe: an "
        "initial run on speaker 2 (female), the cleanest by a loud/quiet-frame SNR-like proxy "
        "(50.7 dB), and the final submitted run on speaker 3 (male), who has the most usable "
        "short-clip audio of the four after filtering to clips of 20 seconds or less (0.58 h, 28% "
        "of his 2.1 h, versus 0.38 h / 14% for speaker 2) despite a lower SNR-like figure (43.6 "
        "dB) -- more short-clip training data outweighed the SNR gap. Most of the raw 8.9 hours "
        "sits in clips much longer than 20 s (VITS trains on short clips), so both runs relied on "
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
        "Computing environment: Google Colab, single NVIDIA T4 GPU. Training writes checkpoints "
        "directly to Google Drive so a Colab disconnect does not lose progress; converted "
        "checkpoints are cached there too so a runtime reset does not require re-downloading the "
        "base model."
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
        "Speaker 2 (pilot run), 20 held-out test-split prompts (scripts/eval_synthesis.py, seed 0): "
        "mean CER 0.211, mean WER 0.654. Vowel-harmony confusions in the ASR transcript, counted "
        "against the text the model was asked to say: e→ɛ 4, ɛ→e 6, o→ɔ 7, "
        "ɔ→o 6 (23 confusions total)."
    )
    doc.add_paragraph(
        "Speaker 3 (final submitted checkpoint), same 20 held-out test-split prompts and procedure: "
        "mean CER 0.235, mean WER 0.617. Vowel-harmony confusions: e→ɛ 2, "
        "ɛ→e 0, o→ɔ 2, ɔ→o 3 (7 total). CER is slightly higher "
        "but WER is lower, and vowel-harmony confusions drop sharply (7 vs. 23) -- consistent with "
        "speaker 3's larger pool of usable short-clip training audio, though with only 20 prompts "
        "this is not a statistically rigorous comparison. WER being substantially higher than CER "
        "for both speakers is expected for a tonal, agglutinative-leaning orthography scored "
        "word-by-word -- a single wrong vowel can flip a whole word while leaving most characters "
        "correct. As noted above, facebook/mms-1b-all is itself an imperfect ASR model for Akan, so "
        "these numbers are an upper bound on the audio's true error, useful for ranking and "
        "spotting outliers rather than as an absolute score. Full per-prompt results for speaker 3 "
        "are in outputs/eval_speaker3_kaggle/eval_report.tsv."
    )

    doc.add_paragraph("3.2 Native-speaker listening test", style="Heading 2")
    doc.add_paragraph(
        "To cover naturalness and tone, three native speakers completed the tasks in "
        "docs/native_speaker_brief.md. Task 1 sanity-checked the natural recordings and a "
        "zero-shot (untrained) baseline, and identified the speaker's variety. Task 2 blind-rated "
        "24 anonymized clips (8 natural, 8 zero-shot baseline, 8 fine-tuned, from the same 8 "
        "held-out sentences) on naturalness and intelligibility (1-5 each) and logged mispronounced "
        "words, paying particular attention to tone and vowel harmony. Task 3 produced new unseen "
        "test sentences in the natural speaker's variety."
    )
    doc.add_paragraph(
        "Note: the clips raters hear for the \"fine-tuned\" system in Task 2 are from the speaker-2 "
        "checkpoint -- the brief and samples were distributed before the speaker-3 retrain above "
        "was run. The listening-test results below therefore speak to the fine-tuning recipe in "
        "general (and to speaker 2 specifically), not to the speaker-3 checkpoint that is the final "
        "submitted model and demo. We flag rather than block on this: re-running Task 2 against "
        "speaker-3 samples is the obvious follow-up but was not possible within this submission's "
        "time."
    ).italic = True
    add_pending(doc, "Task 1 answers (confirmed variety, spelling consistency, audio quality)")
    add_pending(doc, "Task 2 mean naturalness and intelligibility per system (natural / zero-shot "
                      "/ fine-tuned), plus notable mispronunciation / tone / vowel-harmony findings")

    # --- Discussion and limitations -----------------------------------------------
    add_heading(doc, "4. Discussion and Limitations", level=1)
    add_pending(doc, "characterize failure modes once Task 2 results are in: which words/sounds "
                      "the fine-tuned model mispronounces, tone errors flagged by raters, and any "
                      "acoustic artifacts (clicks, buzzing, cut-off words) they reported")
    doc.add_paragraph("Known limitations, independent of the pending results above:", style="Heading 2")
    add_bullets(doc, [
        "Tone is not marked in this orthography and so is not modeled explicitly; the model can "
        "only learn tone patterns indirectly from audio, and we can only evaluate it through human "
        "listening, not automated metrics.",
        "ɛ and ɔ pronunciation depends entirely on correct spelling surviving "
        "normalization; our normalize() maps look-alike and mis-encoded characters (Greek "
        "ε/ς, ↄ) onto the correct letters, but any source text with other encodings "
        "would need the same treatment.",
        "Single speaker only; no multi-speaker or voice-cloning support, and only ~2 hours of "
        "training audio, which is small for TTS fine-tuning.",
        "Digits are not expanded to words; input text containing numbers must already spell them "
        "out.",
        "The base model's licence (CC-BY-NC-4.0) restricts the fine-tuned checkpoint to "
        "non-commercial use.",
        "Three raters: Task 2 results are reported per-rater as well as averaged; we do not claim "
        "statistical significance.",
        "The native-speaker listening test was run against speaker-2 samples before the speaker-3 "
        "retrain; naturalness and tone for the final submitted (speaker-3) checkpoint are only "
        "indirectly evidenced, through the automated vowel-harmony metric above.",
    ])
    doc.add_paragraph("Suggested improvements:", style="Heading 2")
    add_bullets(doc, [
        "More training audio and a second speaker, to test whether quality or variety "
        "generalization improves.",
        "An explicit tone-marking or tone-prediction step ahead of synthesis, evaluated directly "
        "against the native-speaker tone judgments rather than inferred from naturalness scores.",
        "A text-normalization pass for digit expansion, so numeric input does not need to be "
        "pre-written as words.",
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
        "Earlier speaker-2 checkpoint, reported above for comparison: "
        "https://huggingface.co/Lhue-Lhue/akan-tts-mms-vits",
        "Interactive demo (Gradio): scripts/demo_app.py, space/app.py",
    ])
    add_pending(doc, "shared folder link (audio samples + text prompts + checkpoint, accessible "
                      "without a request) for the submission form")

    out = Path(__file__).resolve().parents[1] / "docs" / "report.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    build()

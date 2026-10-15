"""Gradio demo: type Akan (Twi) text, hear it spoken by a fine-tuned model. No external TTS service.

    python scripts/demo_app.py
    python scripts/demo_app.py --model <checkpoint folder or Hub id>   # adds a third, pre-selected option

A dropdown switches between the two published checkpoints (speaker 3, the final retrain, and speaker 2,
the earlier pilot run) so they can be compared side by side. Only one model is held in memory at a time --
switching the dropdown unloads the previous one and loads the new one, which takes a few seconds.

Same text normalization and inference path as synthesize.py, so what you hear here matches what the
evaluation and the native-speaker samples heard. Characters outside the model's vocabulary (including
digits -- numbers are not expanded) are dropped with a warning shown in the UI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import to_inference_text  # noqa: E402

MODEL_CHOICES = {
    "Speaker 3 (final retrain)": "Lhue-Lhue/akan-tts-speaker3-vits",
    "Speaker 2 (earlier pilot run)": "Lhue-Lhue/akan-tts-mms-vits",
}
DEFAULT_LABEL = "Speaker 3 (final retrain)"

EXAMPLES = [
    "Ɛyɛ dɛ yie.",
    "Me kɔ fie.",
    "Akwaaba!",
]

# White background, blue-black accents -- fixed regardless of the viewer's system theme.
_INK = "#0B1226"
_INK_HOVER = "#1C2B4A"
_WHITE = "#FFFFFF"
_PANEL = "#F4F6FA"
_BORDER = "#D8DEE9"


def _theme(gr):
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill=_WHITE,
        body_background_fill_dark=_WHITE,
        background_fill_primary=_WHITE,
        background_fill_primary_dark=_WHITE,
        background_fill_secondary=_PANEL,
        background_fill_secondary_dark=_PANEL,
        block_background_fill=_WHITE,
        block_background_fill_dark=_WHITE,
        block_border_color=_BORDER,
        block_border_color_dark=_BORDER,
        block_label_background_fill=_WHITE,
        block_label_background_fill_dark=_WHITE,
        block_label_text_color=_INK,
        block_label_text_color_dark=_INK,
        block_title_text_color=_INK,
        block_title_text_color_dark=_INK,
        body_text_color=_INK,
        body_text_color_dark=_INK,
        body_text_color_subdued="#4A5568",
        body_text_color_subdued_dark="#4A5568",
        border_color_primary=_BORDER,
        border_color_primary_dark=_BORDER,
        code_background_fill=_PANEL,
        code_background_fill_dark=_PANEL,
        button_primary_background_fill=_INK,
        button_primary_background_fill_dark=_INK,
        button_primary_background_fill_hover=_INK_HOVER,
        button_primary_background_fill_hover_dark=_INK_HOVER,
        button_primary_text_color=_WHITE,
        button_primary_text_color_dark=_WHITE,
        button_secondary_background_fill=_WHITE,
        button_secondary_background_fill_dark=_WHITE,
        button_secondary_border_color=_INK,
        button_secondary_border_color_dark=_INK,
        button_secondary_text_color=_INK,
        button_secondary_text_color_dark=_INK,
        input_background_fill=_WHITE,
        input_background_fill_dark=_WHITE,
        input_border_color=_BORDER,
        input_border_color_dark=_BORDER,
        slider_color=_INK,
        slider_color_dark=_INK,
    )


def build_demo(models: dict, default_label: str):
    import gradio as gr
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only one checkpoint is kept in memory at a time: switching the dropdown drops the old
    # tokenizer/model before loading the new one, so comparing speakers doesn't double the RAM use.
    _cache = {"label": None, "tok": None, "model": None}

    def _get(label):
        if _cache["label"] != label:
            path = models[label]
            tok = AutoTokenizer.from_pretrained(path)
            model = VitsModel.from_pretrained(path).to(device).eval()
            _cache.update(label=label, tok=tok, model=model)
        return _cache["tok"], _cache["model"]

    def speak(text, label, seed, speaking_rate):
        text = (text or "").strip()
        if not text:
            return None, "Type something first."
        tok, model = _get(label)
        # get_vocab() rebuilds via range(vocab_size) and undercounts when the vocab file has a gap
        # (ours reserves id 30 for <unk>, which pushes the last id, "!", out of range) -- union with
        # the raw encoder so every id the tokenizer actually accepts is recognized here too.
        vocab = set(tok.get_vocab()) | set(getattr(tok, "encoder", {}))
        clean, removed = to_inference_text(text, vocab)
        if not clean:
            return None, "Nothing left to say after cleanup (no recognized characters)."
        model.speaking_rate = speaking_rate
        torch.manual_seed(int(seed))
        with torch.no_grad():
            wav = model(**tok(clean, return_tensors="pt").to(device)).waveform[0].cpu().numpy()
        sr = model.config.sampling_rate
        note = f"Speaking ({label}): {clean!r}"
        if removed:
            note += f"  (dropped, not in the model's vocabulary: {removed})"
        return (sr, wav), note

    with gr.Blocks(title="Akan (Twi) TTS demo", theme=_theme(gr)) as demo:
        gr.Markdown(
            "# Akan (Twi) text-to-speech demo\n"
            "Fine-tuned from `facebook/mms-tts-aka` (CC-BY-NC-4.0, non-commercial). Not a "
            "translator -- type Akan text directly. Numbers must be written as words; tone is not "
            "marked in this orthography. Switch the model below to compare speakers; the first "
            "request after switching takes a few seconds longer while it loads."
        )
        text = gr.Textbox(label="Akan text", lines=3, placeholder="Twerɛ Twi wɔ ha...")
        model_choice = gr.Dropdown(label="Model", choices=list(models.keys()), value=default_label)
        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            rate = gr.Slider(label="Speaking rate", minimum=0.5, maximum=1.5, value=1.0, step=0.05)
        btn = gr.Button("Speak", variant="primary")
        audio = gr.Audio(label="Output", type="numpy")
        note = gr.Markdown()
        btn.click(speak, inputs=[text, model_choice, seed, rate], outputs=[audio, note])
        gr.Examples(examples=EXAMPLES, inputs=text)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="optional extra checkpoint folder or Hub id, pre-selected by default")
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = ap.parse_args()

    models = dict(MODEL_CHOICES)
    default_label = DEFAULT_LABEL
    if args.model:
        default_label = f"Custom ({args.model})"
        models[default_label] = args.model

    demo = build_demo(models, default_label)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

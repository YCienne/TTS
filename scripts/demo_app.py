"""Gradio demo: type Akan (Twi) text, hear it spoken by a fine-tuned model. No external TTS service.

    python scripts/demo_app.py
    python scripts/demo_app.py --model <checkpoint folder or Hub id>   # override the default checkpoint

Speaks with the speaker-3 checkpoint (the final retrain) by default -- see the project README for
why speaker 2, an earlier pilot run with less and noisier training audio, was dropped from the demo.
Output is denoised and peak-normalized (akantts.postprocess.clean_audio) to reduce the high-frequency
buzz a checkpoint fine-tuned on under an hour of audio leaves in; this improves how clean the signal
sounds, not pronunciation or prosody.

Same text normalization and inference path as synthesize.py, so what you hear here matches what the
evaluation and the native-speaker samples heard. Characters outside the model's vocabulary (including
digits -- numbers are not expanded) are dropped with a warning shown in the UI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import to_inference_text  # noqa: E402
from akantts.postprocess import clean_audio  # noqa: E402

DEFAULT_MODEL = "Lhue-Lhue/akan-tts-speaker3-vits"

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


def build_demo(model_path: str):
    import gradio as gr
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Loaded lazily on first request, not at server start, so startup stays fast and light.
    _cache = {"tok": None, "model": None}

    def _get():
        if _cache["model"] is None:
            tok = AutoTokenizer.from_pretrained(model_path)
            model = VitsModel.from_pretrained(model_path).to(device).eval()
            _cache.update(tok=tok, model=model)
        return _cache["tok"], _cache["model"]

    def speak(text, seed, speaking_rate):
        text = (text or "").strip()
        if not text:
            return None, "Type something first."
        tok, model = _get()
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
        wav = clean_audio(wav, sr)
        note = f"Speaking: {clean!r}"
        if removed:
            note += f"  (dropped, not in the model's vocabulary: {removed})"
        return (sr, wav), note

    with gr.Blocks(title="Akan (Twi) TTS demo", theme=_theme(gr)) as demo:
        gr.Markdown(
            "# Akan (Twi) text-to-speech demo\n"
            "Fine-tuned from `facebook/mms-tts-aka` (CC-BY-NC-4.0, non-commercial). Not a "
            "translator -- type Akan text directly. Numbers must be written as words; tone is not "
            "marked in this orthography."
        )
        text = gr.Textbox(label="Akan text", lines=3, placeholder="Twerɛ Twi wɔ ha...")
        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            rate = gr.Slider(label="Speaking rate", minimum=0.5, maximum=1.5, value=1.0, step=0.05)
        btn = gr.Button("Speak", variant="primary")
        audio = gr.Audio(label="Output", type="numpy")
        note = gr.Markdown()
        btn.click(speak, inputs=[text, seed, rate], outputs=[audio, note])
        gr.Examples(examples=EXAMPLES, inputs=text)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint folder or Hub id")
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = ap.parse_args()

    demo = build_demo(args.model)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

"""Gradio Space: type Akan (Twi) text, hear it spoken by the fine-tuned model. No external TTS service.

Loads Lhue-Lhue/akan-tts-mms-vits (fine-tuned from facebook/mms-tts-aka) from the Hub by default.
Standalone deployment of scripts/demo_app.py from https://github.com/YCienne/TTS -- same text
normalization and inference path, so what you hear here matches what the evaluation and the
native-speaker samples heard. Characters outside the model's vocabulary (including digits --
numbers are not expanded) are dropped with a warning shown in the UI.
"""
import argparse

from normalize import to_inference_text

DEFAULT_MODEL = "Lhue-Lhue/akan-tts-mms-vits"

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
    tok = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    vocab = set(tok.get_vocab())

    def speak(text, seed, speaking_rate):
        text = (text or "").strip()
        if not text:
            return None, "Type something first."
        clean, removed = to_inference_text(text, vocab)
        if not clean:
            return None, "Nothing left to say after cleanup (no recognized characters)."
        model.speaking_rate = speaking_rate
        torch.manual_seed(int(seed))
        with torch.no_grad():
            wav = model(**tok(clean, return_tensors="pt").to(device)).waveform[0].cpu().numpy()
        sr = model.config.sampling_rate
        note = f"Speaking: {clean!r}"
        if removed:
            note += f"  (dropped, not in the model's vocabulary: {removed})"
        return (sr, wav), note

    with gr.Blocks(title="Akan (Twi) TTS demo", theme=_theme(gr)) as demo:
        gr.Markdown(
            "# Akan (Twi) text-to-speech demo\n"
            f"Model: [`{model_path}`](https://huggingface.co/{model_path}). Fine-tuned from "
            "`facebook/mms-tts-aka` (CC-BY-NC-4.0, non-commercial). Not a translator -- type Akan "
            "text directly. Numbers must be written as words; tone is not marked in this "
            "orthography. Code: [YCienne/TTS](https://github.com/YCienne/TTS)."
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
    args = ap.parse_args()

    demo = build_demo(args.model)
    demo.launch()


if __name__ == "__main__":
    main()

"""Gradio Space: type Akan (Twi) text, hear it spoken by a fine-tuned model. No external TTS service.

Speaks with the speaker-3 checkpoint (the final submitted model -- see the project README for the
speaker-2 vs. speaker-3 tradeoff: speaker 3 has less audible artifact and far fewer vowel-harmony
confusions, speaker 2 has a slightly lower character error rate and drops fewer whole words on
some sentences). Output is denoised
and peak-normalized (akantts.postprocess.clean_audio) to reduce the high-frequency buzz a checkpoint
fine-tuned on under an hour of audio leaves in; this improves how clean the signal sounds, not
pronunciation or prosody. A handful of rare words (letter clusters seen once or not at all in the
~1h training corpus) will still mispronounce regardless of cleanup. Standalone deployment of
scripts/demo_app.py from
https://github.com/YCienne/TTS -- same text normalization and inference path, so what you hear
here matches what the evaluation and the native-speaker samples heard. Characters outside the
model's vocabulary (including digits -- numbers are not expanded) are dropped with a warning shown
in the UI.
"""
import argparse

from normalize import to_inference_text
from postprocess import clean_audio

DEFAULT_MODEL = "Lhue-Lhue/akan-tts-speaker3-vits"

EXAMPLES = [
    "Ɛyɛ dɛ yie.",
    "Me kɔ fie.",
    "Aprɛ no asɛe ama aboa akɔhyɛm.",
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


_CUSTOM_CSS = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
#hero {
    background: linear-gradient(135deg, #0B1226 0%, #1C2B4A 100%);
    border-radius: 18px; padding: 30px 34px; margin-bottom: 4px; color: #FFFFFF;
    box-shadow: 0 8px 24px rgba(11, 18, 38, 0.18);
}
#hero h1 { margin: 0 0 8px 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.01em; color: #FFFFFF; }
#hero p { margin: 0; color: #C7D0E3; font-size: 0.98rem; line-height: 1.55; }
#hero p a { color: #FCD116; }
#hero code {
    color: #FCD116; background: rgba(255, 255, 255, 0.08); padding: 1px 6px; border-radius: 4px;
}
#accent-bar {
    height: 5px; border-radius: 999px; margin: 14px 0 22px 0;
    background: linear-gradient(90deg, #CE1126 0%, #FCD116 34%, #006B3F 68%, #0B1226 100%);
}
#speak-btn { font-size: 1.05rem !important; font-weight: 600 !important; padding: 12px !important; }
#footer-note {
    margin-top: 18px; padding-top: 14px; border-top: 1px solid #D8DEE9;
    color: #6B7280; font-size: 0.85rem; text-align: center; line-height: 1.6;
}
#footer-note a { color: #0B1226; font-weight: 600; text-decoration: none; }
#footer-note a:hover { text-decoration: underline; }
"""


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

    with gr.Blocks(title="Akan (Twi) TTS demo") as demo:
        gr.HTML(
            '<div id="hero"><h1>Akwaaba \U0001F44B — Akan (Twi) Text-to-Speech</h1>'
            "<p>Fine-tuned from <code>facebook/mms-tts-aka</code> (CC-BY-NC-4.0, non-commercial). "
            "Not a translator -- type Akan text directly. Numbers must be written as words; tone "
            "is not marked in this orthography. "
            '<a href="https://github.com/YCienne/TTS" target="_blank">Code: YCienne/TTS</a>.'
            "</p></div>"
        )
        gr.HTML('<div id="accent-bar"></div>')
        with gr.Group():
            text = gr.Textbox(label="\U0001F4DD Akan text", lines=3, placeholder="Twerɛ Twi wɔ ha...")
            btn = gr.Button("\U0001F50A Speak", variant="primary", elem_id="speak-btn")
            audio = gr.Audio(label="Output", type="numpy")
            note = gr.Markdown()
        with gr.Accordion("⚙️ Advanced settings", open=False):
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0)
                rate = gr.Slider(label="Speaking rate", minimum=0.5, maximum=1.5, value=1.0, step=0.05)
        btn.click(speak, inputs=[text, seed, rate], outputs=[audio, note])
        gr.Markdown("**Try an example**")
        gr.Examples(examples=EXAMPLES, inputs=text)
        gr.HTML(
            '<div id="footer-note">Built for the University of Ghana HCI Lab TTS task &middot; '
            '<a href="https://github.com/YCienne/TTS" target="_blank">Source on GitHub</a> &middot; '
            '<a href="https://huggingface.co/Lhue-Lhue/akan-tts-speaker3-vits" target="_blank">'
            "Model on Hugging Face</a></div>"
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint folder or Hub id")
    args = ap.parse_args()

    import gradio as gr

    demo = build_demo(args.model)
    demo.launch(theme=_theme(gr), css=_CUSTOM_CSS)


if __name__ == "__main__":
    main()

"""Gradio demo: type Akan (Twi) text, hear it spoken by the fine-tuned model. No external TTS service.

    python scripts/demo_app.py --model <checkpoint folder or Hub id>

Same text normalization and inference path as synthesize.py, so what you hear here matches what the
evaluation and the native-speaker samples heard. Characters outside the model's vocabulary (including
digits -- numbers are not expanded) are dropped with a warning shown in the UI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akantts.normalize import to_inference_text  # noqa: E402

EXAMPLES = [
    "Ɛyɛ dɛ yie.",
    "Me kɔ fie.",
    "Akwaaba!",
]


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

    with gr.Blocks(title="Akan (Twi) TTS demo") as demo:
        gr.Markdown(
            "# Akan (Twi) text-to-speech demo\n"
            f"Model: `{model_path}`. Fine-tuned from `facebook/mms-tts-aka` "
            "(CC-BY-NC-4.0, non-commercial). Not a translator -- type Akan text directly. "
            "Numbers must be written as words; tone is not marked in this orthography."
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
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint folder or Hub id")
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = ap.parse_args()

    demo = build_demo(args.model)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

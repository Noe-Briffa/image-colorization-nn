import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

from YT.inference_colorize import colorize_image, load_generator, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "unet_colorization_119.pt"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=4)
def get_model(checkpoint: str, features: int, device_name: str):
    device = resolve_device(device_name)
    model = load_generator(Path(checkpoint), features, device)
    return model, device


def run_colorization(
    image: Image.Image | None,
    checkpoint: str,
    saturation: float,
    device_name: str,
    features: int,
) -> Image.Image:
    if image is None:
        raise gr.Error("Ajoute une image avant de lancer la colorisation.")
    try:
        model, device = get_model(str(resolve_project_path(checkpoint).resolve()), int(features), device_name)
        return colorize_image(model, image, device, max_size=1024, boost_saturation=saturation)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_demo(checkpoint: Path, device_name: str, features: int) -> gr.Blocks:
    with gr.Blocks(title="Colorisation d'images — U-Net") as demo:
        gr.Markdown(
            "# Colorisation d'images par U-Net\n"
            "Les couleurs produites sont plausibles, pas une restauration historique certaine."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Image noir et blanc")
            output_image = gr.Image(type="pil", label="Image colorisée")
        with gr.Accordion("Réglages", open=False):
            checkpoint_input = gr.Textbox(value=str(checkpoint), label="Checkpoint")
            saturation = gr.Slider(0.5, 2.0, value=1.2, step=0.05, label="Saturation")
            device = gr.Dropdown(("auto", "cuda", "cpu"), value=device_name, label="Device")
            feature_count = gr.Number(value=features, precision=0, label="Features")
        run_button = gr.Button("Coloriser", variant="primary")
        run_button.click(
            run_colorization,
            inputs=(input_image, checkpoint_input, saturation, device, feature_count),
            outputs=output_image,
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance la démo Gradio locale.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_project_path(args.checkpoint)
    demo = build_demo(args.checkpoint, args.device, args.features)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()

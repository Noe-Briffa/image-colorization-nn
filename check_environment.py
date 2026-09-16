import argparse
import platform
from pathlib import Path

import gradio
import kornia
import lmdb
import lpips
import torch

from YT.inference_colorize import load_generator, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vérifie l'environnement du projet.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_colorization_119.pt",
    )
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_project_path(args.checkpoint)
    device = resolve_device(args.device)
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision CUDA: {torch.version.cuda}")
    print(f"Kornia: {kornia.__version__}")
    print(f"Gradio: {gradio.__version__}")
    print(f"LMDB: {lmdb.__version__}")
    print(f"LPIPS: {getattr(lpips, '__version__', '0.1.4')}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device sélectionné: {device}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: {args.checkpoint}")
    load_generator(args.checkpoint, args.features, device)
    print(f"Checkpoint compatible: {args.checkpoint.name}")


if __name__ == "__main__":
    main()

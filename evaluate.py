import argparse
import json
import math
import random
from pathlib import Path

import kornia
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw

from unet_learning import LazyLabDatasetLMDB, lab01_to_rgb01_fast
from YT.inference_colorize import load_generator, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_indices(dataset_size: int, max_samples: int, seed: int) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("Le dataset est vide.")
    if max_samples <= 0:
        raise ValueError("--max-samples doit être supérieur à zéro.")
    count = min(dataset_size, max_samples)
    return sorted(random.Random(seed).sample(range(dataset_size), count))


def tensor_to_image(rgb01: torch.Tensor) -> Image.Image:
    array = (
        rgb01.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    return Image.fromarray(array)


def save_comparison(
    luminance: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    output_path: Path,
) -> None:
    gray = luminance.detach().cpu().clamp(0, 1).repeat(1, 3, 1, 1)
    panels = [tensor_to_image(gray), tensor_to_image(prediction), tensor_to_image(reference)]
    labels = ("Entrée N&B", "Prédiction", "Référence")
    width, height = panels[0].size
    header = 28
    canvas = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (position * width, header))
        draw.text((position * width + 8, 7), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = resolve_device(args.device)
    dataset = LazyLabDatasetLMDB(str(args.dataset))
    indices = select_indices(len(dataset), args.max_samples, args.seed)
    generator = load_generator(args.checkpoint, args.features, device)
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()

    collected = {"psnr": [], "ssim": [], "delta_e": [], "lpips": []}
    args.output.mkdir(parents=True, exist_ok=True)

    for position, index in enumerate(indices):
        luminance, ab_reference = dataset[index]
        luminance = luminance.unsqueeze(0).to(device)
        ab_reference = ab_reference.unsqueeze(0).to(device)
        ab_prediction = ((generator(luminance) + 1.0) / 2.0).clamp(0, 1)
        rgb_prediction = lab01_to_rgb01_fast(luminance, ab_prediction).clamp(0, 1)
        rgb_reference = lab01_to_rgb01_fast(luminance, ab_reference).clamp(0, 1)

        mse = torch.mean((rgb_prediction - rgb_reference) ** 2).item()
        collected["psnr"].append(10 * math.log10(1 / mse) if mse > 0 else float("inf"))
        collected["ssim"].append(
            kornia.metrics.ssim(rgb_prediction, rgb_reference, 11).mean().item()
        )
        lab_prediction = kornia.color.rgb_to_lab(rgb_prediction)
        lab_reference = kornia.color.rgb_to_lab(rgb_reference)
        collected["delta_e"].append(
            torch.linalg.norm(lab_prediction - lab_reference, dim=1).mean().item()
        )
        collected["lpips"].append(
            perceptual(rgb_prediction * 2 - 1, rgb_reference * 2 - 1).mean().item()
        )

        if position < 6:
            save_comparison(
                luminance,
                rgb_prediction,
                rgb_reference,
                args.output / f"comparison_{position + 1:02d}_idx{index}.png",
            )

    report = {
        "protocol": "held-out-lmdb-v1",
        "checkpoint": args.checkpoint.name,
        "dataset": args.dataset.name,
        "seed": args.seed,
        "sample_count": len(indices),
        "selected_indices": indices,
        "device": str(device),
        "metrics": {name: summarize(values) for name, values in collected.items()},
    }
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue un checkpoint sur un LMDB tenu à part.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "evaluation")
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("dataset", "checkpoint", "output"):
        setattr(args, name, resolve_project_path(getattr(args, name)))
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

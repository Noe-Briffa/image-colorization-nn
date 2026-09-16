import argparse
import os
import re
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import kornia.color as K

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet_learning import UNet, lab01_to_rgb01_fast


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_latest_checkpoint(ckpt_dir: str, base_name: str) -> str:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Dossier checkpoints introuvable: {ckpt_dir}")

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.pt$")
    candidates = []
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            candidates.append((int(m.group(1)), os.path.join(ckpt_dir, f)))

    if not candidates:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé dans {ckpt_dir} avec le pattern {base_name}_<epoch>.pt"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé, mais aucune carte CUDA n'est disponible.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_generator(checkpoint_path: str | Path, features: int, device: torch.device) -> UNet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or "G" not in ckpt:
        raise ValueError("Checkpoint incompatible: clé 'G' absente.")

    G = UNet(in_channels=1, out_channels=2, features=features).to(device=device)
    try:
        G.load_state_dict(ckpt["G"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint incompatible avec cette architecture. "
            f"Vérifie --features (valeur actuelle: {features})."
        ) from exc

    G.eval()
    return G


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """
    PIL RGB -> torch float tensor [1,3,H,W] en 0..1
    (sans dépendre de torchvision, et compatible toutes versions Kornia)
    """
    arr = np.array(img, dtype=np.uint8)          # [H,W,3]
    t = torch.from_numpy(arr).permute(2, 0, 1)   # [3,H,W]
    t = t.unsqueeze(0).float() / 255.0           # [1,3,H,W] en 0..1
    return t


@torch.no_grad()
def colorize_image(
    G: UNet,
    img_pil: Image.Image,
    device: torch.device,
    max_size: int,
    boost_saturation: float,
) -> Image.Image:
    img = img_pil.convert("RGB")

    # Resize si trop grande
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    # PIL -> tensor 0..1
    x = pil_to_tensor01(img).to(device, memory_format=torch.channels_last)

    # multiple de 16 (U-Net downsample)
    _, _, H, W = x.shape
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    # RGB -> Lab
    Lab = K.rgb_to_lab(x)
    L = Lab[:, 0:1, :, :]  # 0..100
    L01 = (L / 100.0).clamp(0.0, 1.0)

    # Prédiction ab en [-1,1] -> [0,1]
    ab_fake = G(L01)
    ab01 = ((ab_fake + 1.0) / 2.0).clamp(0.0, 1.0)

    # Boost saturation optionnel
    if boost_saturation != 1.0:
        ab01 = ((ab01 - 0.5) * boost_saturation + 0.5).clamp(0.0, 1.0)

    # Lab01 -> RGB01
    rgb01 = lab01_to_rgb01_fast(L01, ab01).clamp(0.0, 1.0)

    # Unpad
    if pad_h or pad_w:
        rgb01 = rgb01[:, :, :H, :W]

    # tensor -> PIL
    rgb_u8 = (rgb01[0].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(rgb_u8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorise un dossier d'images avec un checkpoint U-Net.")
    parser.add_argument("--input", type=Path, required=True, help="Dossier d'images à coloriser.")
    parser.add_argument("--output", type=Path, required=True, help="Dossier de sortie.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .pt à utiliser.")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--features", type=int, default=48, help="Largeur de base du U-Net.")
    parser.add_argument("--max-size", type=int, default=1024, help="Plus grand côté en pixels.")
    parser.add_argument("--saturation", type=float, default=1.2, help="Facteur de saturation, 1.0 désactive.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = resolve_project_path(args.input)
    args.output = resolve_project_path(args.output)
    args.checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    if args.checkpoint is not None:
        args.checkpoint = resolve_project_path(args.checkpoint)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    ckpt_path = str(args.checkpoint) if args.checkpoint else find_latest_checkpoint(
        str(args.checkpoint_dir), "unet_colorization"
    )
    print(f"Checkpoint: {ckpt_path}")

    G = load_generator(ckpt_path, args.features, device)

    in_dir = args.input
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"INPUT_DIR invalide: {in_dir.resolve()}")

    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print(f"Aucune image trouvee dans {in_dir.resolve()} (ext: {sorted(EXTS)})")
        return

    ok, fail = 0, 0
    for i, img_path in enumerate(files, start=1):
        try:
            with Image.open(img_path) as img:
                out_img = colorize_image(G, img, device, args.max_size, args.saturation)

            out_path = out_dir / img_path.name
            out_img.save(out_path)

            ok += 1
            print(f"[{i}/{len(files)}] OK {img_path.name} -> {out_path}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] ERREUR {img_path.name} | {type(e).__name__}: {e}")

    print(f"\nTerminé: {ok} ok, {fail} échecs. Sortie: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

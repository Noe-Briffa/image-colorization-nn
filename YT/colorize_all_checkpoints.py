import os
import re
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import kornia.color as K

from unet_learning import UNet, lab01_to_rgb01_fast


# =========================
# CONFIG À MODIFIER ICI
# =========================
INPUT_DIR = "../in_bw"  # dossier contenant les images N&B (ou RGB désaturées)
OUTPUT_DIR = "../output_colorise_all"  # dossier de sortie global

CHECKPOINT_DIR = "../checkpoints"
CHECKPOINT_BASENAME = "unet_colorization"  # ex: unet_colorization_119.pt

FEATURES = 48
MAX_SIZE = 1024
BOOST_SATURATION = 1.2  # 1.0 = off

# Optionnel: restreindre les epochs exportées (None = tout)
EPOCH_MIN = None  # ex: 1
EPOCH_MAX = None  # ex: 120

# Optionnel: exporter seulement 1 frame sur N pour accélérer (1 = tout)
EXPORT_EVERY_N_EPOCHS = 1  # ex: 2 => 1,3,5,...

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_all_checkpoints(ckpt_dir: str, base_name: str):
    """Retourne une liste triée [(epoch:int, path:str), ...]"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Dossier checkpoints introuvable: {ckpt_dir}")

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.pt$")
    candidates = []
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            epoch = int(m.group(1))
            path = os.path.join(ckpt_dir, f)
            candidates.append((epoch, path))

    if not candidates:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé dans {ckpt_dir} avec le pattern {base_name}_<epoch>.pt"
        )

    candidates.sort(key=lambda x: x[0])

    # Filtrage optionnel
    if EPOCH_MIN is not None:
        candidates = [c for c in candidates if c[0] >= EPOCH_MIN]
    if EPOCH_MAX is not None:
        candidates = [c for c in candidates if c[0] <= EPOCH_MAX]

    # Sous-échantillonnage optionnel
    if EXPORT_EVERY_N_EPOCHS and EXPORT_EVERY_N_EPOCHS > 1:
        candidates = [c for idx, c in enumerate(candidates) if idx % EXPORT_EVERY_N_EPOCHS == 0]

    return candidates


def load_generator(checkpoint_path: str, features: int, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)

    G = UNet(in_channels=1, out_channels=2, features=features).to(device=device)
    missing, unexpected = G.load_state_dict(ckpt["G"], strict=False)

    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    G.eval()
    return G


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor [1,3,H,W] en 0..1 (sans torchvision)."""
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t.unsqueeze(0).float() / 255.0
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

    x = pil_to_tensor01(img).to(device, memory_format=torch.channels_last)

    # Multiple de 16 (U-Net downsample)
    _, _, H, W = x.shape
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    # RGB -> Lab
    Lab = K.rgb_to_lab(x)
    L = Lab[:, 0:1, :, :]            # 0..100
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

    rgb_u8 = (rgb01[0].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(rgb_u8, mode="RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"INPUT_DIR invalide: {in_dir.resolve()}")

    img_files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not img_files:
        print(f"⚠️ Aucune image trouvée dans {in_dir.resolve()} (ext: {sorted(EXTS)})")
        return

    checkpoints = find_all_checkpoints(CHECKPOINT_DIR, CHECKPOINT_BASENAME)
    print(f"✅ {len(checkpoints)} checkpoints trouvés.")
    print("   Exemples:", checkpoints[:3], "..." if len(checkpoints) > 3 else "")

    # Option: accélération (safe) en AMP si CUDA dispo
    use_amp = (device.type == "cuda")

    total_jobs = len(checkpoints) * len(img_files)
    done = 0
    ok, fail = 0, 0

    for epoch, ckpt_path in checkpoints:
        print(f"\n===== Epoch {epoch} | checkpoint: {ckpt_path} =====")
        G = load_generator(ckpt_path, FEATURES, device)

        for img_path in img_files:
            done += 1
            try:
                with Image.open(img_path) as img:
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out_img = colorize_image(G, img, device, MAX_SIZE, BOOST_SATURATION)
                    else:
                        out_img = colorize_image(G, img, device, MAX_SIZE, BOOST_SATURATION)

                # nomdelimage + _epoch
                out_name = f"{img_path.stem}_{epoch}.png"
                out_path = out_dir / out_name
                out_img.save(out_path)

                ok += 1
                if done % 50 == 0 or done == total_jobs:
                    print(f"[{done}/{total_jobs}] ✅ {out_name}")
            except Exception as e:
                fail += 1
                print(f"[{done}/{total_jobs}] ❌ {img_path.name} (epoch {epoch}) | {type(e).__name__}: {e}")

        # libère un peu (utile si beaucoup de checkpoints)
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTerminé: {ok} ok, {fail} échecs. Sortie: {out_dir.resolve()}")


main()

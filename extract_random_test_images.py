import os
import random
from typing import Any, Tuple, Optional

import torch
import numpy as np
from PIL import Image

# =========================
# CONFIG À MODIFIER ICI
# =========================
TEST_IMAGES_PT = "test_images.pt"
OUTPUT_DIR = "output"
N_SAMPLES = 500
SEED = 42  # change si tu veux un autre tirage


def _try_get_images(obj: Any) -> Any:
    """
    Essaie d'extraire un tenseur d'images depuis des structures courantes :
    - Tensor directement
    - dict avec clés: images, x, X, data
    - tuple/list: (images, labels) etc.
    """
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        for k in ["images", "image", "x", "X", "data", "inputs"]:
            if k in obj:
                return obj[k]
        # parfois c'est nested
        for v in obj.values():
            if torch.is_tensor(v):
                return v

    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        # cas typique: (x, y)
        if torch.is_tensor(obj[0]):
            return obj[0]
        # parfois c'est une liste de tensors
        if all(torch.is_tensor(x) for x in obj):
            return torch.stack(obj, dim=0)

    return None


def _as_uint8_img(t: torch.Tensor) -> np.ndarray:
    """
    Convertit un tensor image en uint8 HxWxC (C=1 ou 3), en essayant d'inférer l'échelle:
    - [0,1] float
    - [-1,1] float
    - [0,255] float/int
    """
    t = t.detach().cpu()

    # gérer images torch (C,H,W) ou (H,W,C) ou (H,W)
    if t.ndim == 2:
        t = t.unsqueeze(0)  # -> (1,H,W)
    elif t.ndim == 3:
        pass
    else:
        raise ValueError(f"Tensor image doit être 2D/3D, reçu shape={tuple(t.shape)}")

    # si c'est (H,W,C), on bascule en (C,H,W)
    if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)

    if t.shape[0] not in (1, 3):
        raise ValueError(f"Canaux attendus 1 ou 3, reçu C={t.shape[0]} shape={tuple(t.shape)}")

    # mettre en float pour scaler
    tf = t.float()

    # inférer échelle
    mn = float(tf.min())
    mx = float(tf.max())

    if mx <= 1.0 and mn >= 0.0:
        tf = tf * 255.0
    elif mn >= -1.0 and mx <= 1.0:
        tf = (tf * 0.5 + 0.5) * 255.0
    else:
        # suppose déjà 0..255 (ou proche)
        pass

    tf = tf.clamp(0, 255).round().byte()

    # (C,H,W) -> (H,W,C)
    arr = tf.permute(1, 2, 0).numpy()
    return arr


def main():
    if not os.path.isfile(TEST_IMAGES_PT):
        raise FileNotFoundError(f"Fichier introuvable: {TEST_IMAGES_PT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    obj = torch.load(TEST_IMAGES_PT, map_location="cpu")
    images = _try_get_images(obj)

    if images is None:
        raise RuntimeError(
            "Impossible d'extraire les images depuis test_images.pt.\n"
            "Formats gérés: Tensor direct, dict{images/x/data}, tuple/list (x,y).\n"
            "Dis-moi le type et les clés, et je te fais l'extraction adaptée."
        )

    if not torch.is_tensor(images):
        raise TypeError(f"Extraction a renvoyé un objet non-tensor: {type(images)}")

    # Attendu: (N,C,H,W) ou (N,H,W,C) ou (N,H,W)
    if images.ndim == 3:
        # (N,H,W) -> (N,1,H,W)
        images = images.unsqueeze(1)
    elif images.ndim != 4:
        raise ValueError(f"Tensor images attendu 3D/4D, reçu shape={tuple(images.shape)}")

    N = images.shape[0]
    if N == 0:
        raise ValueError("Aucune image dans le tensor (N=0).")

    k = min(N_SAMPLES, N)

    random.seed(SEED)
    idxs = random.sample(range(N), k)

    print(f"✅ Dataset: N={N} | Extraction: {k} images -> {OUTPUT_DIR}/")

    for i, idx in enumerate(idxs):
        img_t = images[idx]

        # Si img_t est (H,W,C) au lieu de (C,H,W), _as_uint8_img gère.
        arr = _as_uint8_img(img_t)

        # PIL
        if arr.shape[-1] == 1:
            pil = Image.fromarray(arr[:, :, 0], mode="L")
        else:
            pil = Image.fromarray(arr, mode="RGB")

        out_path = os.path.join(OUTPUT_DIR, f"img_{i:03d}_idx{idx}.png")
        pil.save(out_path)

    print("✅ Terminé.")


main()

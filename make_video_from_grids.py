import re
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
INPUT_DIR = "montage_grid"
OUTPUT_VIDEO = "epochs_grid.mp4"

DURATION_SECONDS = 40      # durée totale souhaitée
FPS = 60                   # <-- TU CONTROLES ICI
HOLD_START = 1.0           # secondes de pause au début
HOLD_END = 1.5             # secondes de pause à la fin

PATTERN = re.compile(r"^grid_epoch_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def list_frames(input_dir: Path):
    frames = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.match(p.name)
        if m:
            epoch = int(m.group(1))
            frames.append((epoch, p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]


def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {in_dir.resolve()}")

    frame_paths = list_frames(in_dir)
    if not frame_paths:
        raise RuntimeError("Aucune frame détectée.")

    n_src = len(frame_paths)

    total_frames = int(DURATION_SECONDS * FPS)
    hold_start_frames = int(HOLD_START * FPS)
    hold_end_frames = int(HOLD_END * FPS)

    transition_frames = total_frames - hold_start_frames - hold_end_frames

    if transition_frames <= 0:
        raise ValueError("Durée trop courte par rapport aux holds.")

    print(f"Frames source: {n_src}")
    print(f"FPS: {FPS}")
    print(f"Durée totale: {DURATION_SECONDS}s ({total_frames} frames)")
    print(f"Hold start: {hold_start_frames} frames")
    print(f"Hold end: {hold_end_frames} frames")
    print(f"Transition: {transition_frames} frames")

    # Lire première image
    first = imageio.imread(frame_paths[0])
    h, w = first.shape[:2]

    with imageio.get_writer(
        OUTPUT_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None
    ) as writer:

        # 1️⃣ HOLD START
        for _ in range(hold_start_frames):
            writer.append_data(first)

        # 2️⃣ TRANSITION
        for i in range(transition_frames):
            # Mapping fluide des frames source
            t = i / transition_frames
            src_idx = int(t * (n_src - 1))
            img = imageio.imread(frame_paths[src_idx])

            # sécurité taille
            if img.shape[0] != h or img.shape[1] != w:
                img = np.array(Image.fromarray(img).resize((w, h), Image.BICUBIC))

            writer.append_data(img)

        # 3️⃣ HOLD END
        last = imageio.imread(frame_paths[-1])
        for _ in range(hold_end_frames):
            writer.append_data(last)

    print(f"\n✅ Vidéo créée : {Path(OUTPUT_VIDEO).resolve()}")


if __name__ == "__main__":
    main()
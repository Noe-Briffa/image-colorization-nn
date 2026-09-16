import os
import re
from pathlib import Path
from PIL import Image

# =========================
# CONFIG
# =========================
INPUT_DIR = "output_colorise_all"   # là où tu as toutes les images nom_epoch.png
OUTPUT_DIR = "montage_grid"        # sortie: une image par epoch

GRID_COLS = 4
GRID_ROWS = 3
EXPECTED_IMAGES_PER_EPOCH = GRID_COLS * GRID_ROWS  # 12

# Si tes images n'ont pas exactement les mêmes dimensions, on les force à une taille uniforme :
CELL_W = None  # ex: 512 (sinon auto: taille de la 1ère image)
CELL_H = None  # ex: 512

# Marges (en pixels)
PAD = 12            # padding entre les images
OUTER_PAD = 20      # marge autour de la grille
BG_COLOR = (10, 10, 10)  # fond gris/noir

# Pattern attendu: "nomdelimage_<epoch>.png"
# (epoch = digits en fin de nom avant l'extension)
FILENAME_RE = re.compile(r"^(?P<base>.+)_(?P<epoch>\d+)\.(?P<ext>png|jpg|jpeg|webp)$", re.IGNORECASE)


def collect_by_epoch(input_dir: Path):
    by_epoch = {}  # epoch -> list[(base, path)]
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        base = m.group("base")
        epoch = int(m.group("epoch"))
        by_epoch.setdefault(epoch, []).append((base, p))
    return by_epoch


def make_grid(image_paths, out_path: Path, cols: int, rows: int):
    # Ouvrir les images
    imgs = [Image.open(p).convert("RGB") for p in image_paths]

    # Déterminer taille cellule
    cell_w = CELL_W or imgs[0].width
    cell_h = CELL_H or imgs[0].height

    # Uniformiser tailles (sinon collage dégueu)
    resized = []
    for im in imgs:
        if im.size != (cell_w, cell_h):
            resized.append(im.resize((cell_w, cell_h), Image.BICUBIC))
        else:
            resized.append(im)

    # Taille finale
    grid_w = OUTER_PAD * 2 + cols * cell_w + (cols - 1) * PAD
    grid_h = OUTER_PAD * 2 + rows * cell_h + (rows - 1) * PAD

    canvas = Image.new("RGB", (grid_w, grid_h), BG_COLOR)

    # Coller
    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = OUTER_PAD + c * (cell_w + PAD)
        y = OUTER_PAD + r * (cell_h + PAD)
        canvas.paste(im, (x, y))

    # Nettoyage
    for im in imgs:
        im.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR introuvable: {in_dir.resolve()}")

    by_epoch = collect_by_epoch(in_dir)

    if not by_epoch:
        raise RuntimeError(
            f"Aucune image reconnue dans {in_dir.resolve()}.\n"
            f"Attendu: nomdelimage_<epoch>.png (ex: cat_12.png)"
        )

    epochs = sorted(by_epoch.keys())
    print(f"✅ {len(epochs)} epochs détectés: {epochs[0]}..{epochs[-1]}")

    # Pour garder un ordre stable des 12 images dans la grille,
    # on trie par base name (alphabetique).
    # Si tu veux un ordre custom (ex: toujours la même disposition), dis-moi et on le fixe.
    for epoch in epochs:
        items = sorted(by_epoch[epoch], key=lambda t: t[0].lower())  # [(base, path), ...]
        if len(items) != EXPECTED_IMAGES_PER_EPOCH:
            print(f"⚠️ Epoch {epoch}: {len(items)} images trouvées (attendu {EXPECTED_IMAGES_PER_EPOCH}). -> skip")
            continue

        paths = [p for _, p in items]
        out_path = out_dir / f"grid_epoch_{epoch:03d}.png"
        make_grid(paths, out_path, GRID_COLS, GRID_ROWS)
        print(f"✅ {out_path.name}")

    print(f"\nTerminé. Dossier: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
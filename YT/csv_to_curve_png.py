import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================
# CONFIG
# ======================

CSV_PATH = "../full_metrics.csv"     # <-- ton fichier
OUT_DIR = "graphs_png"            # dossier de sortie

WIDTH_PX = 1920
HEIGHT_PX = 600
DPI = 100

LINE_WIDTH = 10
LINE_COLOR = "#ff00ff"            # blanc (tu peux changer)
MAD_Z = 6.0                       # plus bas = plus agressif (4-6 recommandé)
ROLL_WIN = 9                      # fenêtre impair (7/9/11)

MAKE_OVERVIEW_GRID = True         # True => image grille de toutes les colonnes


# ======================
# HELPERS
# ======================

def replace_first_with_second(v: np.ndarray) -> np.ndarray:
    v = v.astype(float).copy()
    if len(v) > 1 and np.isfinite(v[1]):
        v[0] = v[1]
    return v


def rolling_median_mad_outlier_smooth(v: np.ndarray, win: int = 9, z: float = 6.0) -> np.ndarray:
    """
    Détecte les outliers par médiane glissante + MAD, puis remplace chaque outlier
    par la moyenne des voisins (i-1 et i+1).
    """
    v = v.astype(float).copy()
    n = len(v)
    if n < 5:
        return v

    # rolling median
    s = pd.Series(v)
    med = s.rolling(window=win, center=True, min_periods=max(3, win // 2)).median()

    # MAD (Median Absolute Deviation) rolling
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window=win, center=True, min_periods=max(3, win // 2)).median()

    # robust z-score approximation: 0.6745 * (x - median) / MAD
    # if MAD==0 => avoid divide
    mad_safe = mad.replace(0, np.nan)
    rz = 0.6745 * (s - med) / mad_safe

    outlier = rz.abs() > z
    outlier = outlier.fillna(False).to_numpy()

    # Replace outliers with mean of neighbors (use nearest finite neighbors)
    for i in range(n):
        if not outlier[i]:
            continue

        left = i - 1
        right = i + 1

        # find finite left
        while left >= 0 and not np.isfinite(v[left]):
            left -= 1

        # find finite right
        while right < n and not np.isfinite(v[right]):
            right += 1

        if left >= 0 and right < n:
            v[i] = 0.5 * (v[left] + v[right])
        elif left >= 0:
            v[i] = v[left]
        elif right < n:
            v[i] = v[right]
        else:
            v[i] = 0.0

    return v


def normalize_01(v: np.ndarray) -> np.ndarray:
    v = v.astype(float).copy()
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    mn = np.min(v[finite])
    mx = np.max(v[finite])
    if mx - mn < 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def export_curve_png(values_norm: np.ndarray, out_path: str, width_px: int, height_px: int, dpi: int):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    plt.plot(values_norm, color=LINE_COLOR, linewidth=LINE_WIDTH)

    # No axes, no padding, transparent background
    plt.axis("off")
    plt.margins(0)
    plt.xlim(0, len(values_norm) - 1)
    plt.ylim(0, 1)

    plt.gca().set_facecolor("none")
    fig.patch.set_alpha(0)

    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ======================
# MAIN
# ======================

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Si ton CSV a une colonne epoch, on prend la 1ère valeur de chaque epoch.
# Sinon, on considère que chaque colonne contient déjà une valeur par "epoch" (ligne).
has_epoch = "epoch" in df.columns

# Colonnes candidates: numériques, hors epoch
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "epoch"]

if not numeric_cols:
    raise ValueError("Aucune colonne numérique trouvée dans le CSV (à part 'epoch').")

saved_paths = []

for col in numeric_cols:
    if has_epoch:
        series = df.groupby("epoch")[col].first()
        v = series.to_numpy(dtype=float)
    else:
        v = df[col].to_numpy(dtype=float)

    if len(v) < 2:
        continue

    # 1) Fix première valeur
    v = replace_first_with_second(v)

    # 2) Outliers -> moyenne des voisins
    v = rolling_median_mad_outlier_smooth(v, win=ROLL_WIN, z=MAD_Z)

    # 3) Normalisation 0-1
    v_norm = normalize_01(v)

    # 4) Export PNG
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in col)
    out_path = os.path.join(OUT_DIR, f"{safe_name}.png")
    export_curve_png(v_norm, out_path, WIDTH_PX, HEIGHT_PX, DPI)

    saved_paths.append(out_path)

print(f"✅ Export terminé : {len(saved_paths)} courbes dans '{OUT_DIR}'")

# Optionnel: overview en grille
if MAKE_OVERVIEW_GRID and saved_paths:
    cols = 3
    rows = int(np.ceil(len(saved_paths) / cols))
    fig = plt.figure(figsize=(cols * 6, rows * 2.5), dpi=150)
    for idx, col in enumerate(numeric_cols):
        if idx >= len(saved_paths):
            break
        ax = plt.subplot(rows, cols, idx + 1)

        # Recompute values_norm for preview (fast)
        if has_epoch:
            v = df.groupby("epoch")[col].first().to_numpy(dtype=float)
        else:
            v = df[col].to_numpy(dtype=float)
        if len(v) < 2:
            continue
        v = replace_first_with_second(v)
        v = rolling_median_mad_outlier_smooth(v, win=ROLL_WIN, z=MAD_Z)
        v_norm = normalize_01(v)

        ax.plot(v_norm, linewidth=1.5)
        ax.set_title(col, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    overview_path = os.path.join(OUT_DIR, "_overview_grid.png")
    fig.savefig(overview_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"🧩 Overview exportée : {overview_path}")

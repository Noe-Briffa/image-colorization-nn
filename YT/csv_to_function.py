import pandas as pd
import numpy as np

# -------- PARAMS --------
csv_path = "../full_metrics.csv"  # <-- mets ton chemin
degree = 4                     # 5 à 8 conseillé

# Colonnes à ignorer (même si numériques)
ignore_cols = {"epoch", "batch"}

# -------- LOAD --------
df = pd.read_csv(csv_path)

if "epoch" not in df.columns:
    raise ValueError("Le CSV doit contenir une colonne 'epoch'.")

# 1ère valeur de chaque epoch (première ligne)
g = df.groupby("epoch", as_index=False).first()

# Garder seulement les colonnes numériques (float/int)
num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ignore_cols]

# Axe x normalisé 0..1 sur N points
N = len(g)
px = np.linspace(0, 1, N)

def poly_to_fusion_px(coeffs: np.ndarray) -> str:
    """coeffs from np.polyfit highest degree -> constant"""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        if abs(c) < 1e-12:
            continue
        if p == 0:
            terms.append(f"{c:.10g}")
        elif p == 1:
            terms.append(f"{c:.10g}*px")
        else:
            terms.append(f"{c:.10g}*px^{p}")
    return " + ".join(terms) if terms else "0"

# -------- FIT & PRINT --------
print("\n=== FUSION EXPRESSIONS (utilise px) ===\n")

for col in num_cols:
    y = g[col].to_numpy(dtype=float)

    # Si colonne quasi-constante ou trop de NaN => skip proprement
    if np.isnan(y).all():
        print(f"\n--- {col} ---\nreturn 0  -- (tout NaN)")
        continue

    # Remplacer NaN (au cas où) par interpolation simple + bords
    s = pd.Series(y).interpolate(limit_direction="both")
    y = s.to_numpy(dtype=float)

    # Fit polynôme sur px in [0,1]
    coeffs = np.polyfit(px, y, degree)
    expr = poly_to_fusion_px(coeffs)

    print(f"\n--- {col} (deg {degree}) ---")
    print(f"return {expr}")

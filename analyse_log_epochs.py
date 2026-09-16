import pandas as pd

CSV_PATH = "checkpoints/full_metrics.csv"   # change si besoin
EXPECTED_BATCHES = 3063

df = pd.read_csv(CSV_PATH)

# Assurer bon tri
df = df.sort_values(by=["epoch", "batch"]).reset_index(drop=True)

epochs = sorted(df["epoch"].unique())


# ================================
# 1) Epochs manquantes
# ================================
min_e, max_e = min(epochs), max(epochs)
all_expected = set(range(min_e, max_e + 1))
epochs_missing = sorted(list(all_expected - set(epochs)))


# ================================
# 2) Epochs dupliquées
# (ex : epoch 14 apparaît plusieurs fois dans des blocs séparés)
# ================================
epoch_counts = df["epoch"].value_counts()
epochs_duplicated = [e for e, c in epoch_counts.items() if c > EXPECTED_BATCHES]


# ================================
# 3) Epochs incomplètes
# ================================
epochs_incomplete = [e for e, c in epoch_counts.items() if c < EXPECTED_BATCHES]


# ================================
# 4) Epochs avec ordre de batch incohérent
# (ex : batch repart à zéro au milieu → redémarrage)
# ================================
epochs_inconsistent = []

for e in epochs:
    df_e = df[df["epoch"] == e]
    batches = df_e["batch"].values

    # détecte si batch diminue à un moment → reset → restart
    if any(batches[i] > batches[i+1] for i in range(len(batches)-1)):
        epochs_inconsistent.append(e)


# ================================
# 5) Résumé propre
# ================================
print("\n==================== ANALYSE DES LOGS ====================\n")

print(f"Total epochs trouvés : {len(epochs)} (min={min_e}, max={max_e})\n")

print("⚠️  Epochs manquantes :", epochs_missing or "Aucune")
print("⚠️  Epochs dupliquées (plus de 3063 batchs) :", epochs_duplicated or "Aucune")
print("⚠️  Epochs incomplètes :", epochs_incomplete or "Aucune")
print("⚠️  Epochs avec ordre de batch incohérent :", epochs_inconsistent or "Aucune")

print("\n===========================================================\n")

# ================================
# 6) Export facultatif : epochs bons uniquement
# ================================
valid_epochs = [
    e for e in epochs
    if e not in epochs_incomplete
    and e not in epochs_duplicated
    and e not in epochs_inconsistent
]

print("✨ Epochs parfaitement propres :", valid_epochs[:5], "... (voir fichier)")

df_clean = df[df["epoch"].isin(valid_epochs)]
df_clean.to_csv("full_metrics_clean.csv", index=False)

print("\n📁 Fichier exporté : full_metrics_clean.csv\n")

import pandas as pd

# === PARAMÈTRES ===
csv_path = "../full_metrics.csv"  # <-- change ici
epoch_column = "epoch"         # nom de la colonne epoch
output_decimals = 6            # nombre de décimales

# === LECTURE CSV ===
df = pd.read_csv(csv_path)

# === GARDE LA PREMIÈRE VALEUR DE CHAQUE EPOCH ===
df_first = df.groupby(epoch_column, as_index=False).first()

# === TRI PAR EPOCH (au cas où) ===
df_first = df_first.sort_values(epoch_column)

# === GÉNÉRATION DES LISTES POUR CHAQUE COLONNE ===
for col in df_first.columns:
    if col == epoch_column:
        continue  # on ignore la colonne epoch

    values = df_first[col].tolist()

    # Nettoyage: convertir en float et formatter
    formatted_values = [
        f"{float(v):.{output_decimals}f}"
        for v in values
        if pd.notna(v)
    ]

    # Format Lua / Fusion
    fusion_list = ",".join(formatted_values)

    print(f"\n=== {col.upper()} ===")
    print(fusion_list)
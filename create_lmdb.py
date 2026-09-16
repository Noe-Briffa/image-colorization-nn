import os
import lmdb
import pickle
import torch
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "dataset/dataset_lab_32_test"
OUTPUT_LMDB = "dataset/dataset_lab_32_test.lmdb"
MAP_SIZE_GB = 1
COMMIT_EVERY = 200  # ultra safe pour Windows
# ============================================================


def tensor_to_bytes(t):
    """Convertit un tensor CPU en bytes + infos shape/dtype (pas de copie inutile)."""
    arr = t.numpy()
    return arr.tobytes(), arr.shape, str(arr.dtype)


def build_lmdb(input_dir: str, output_path: str, map_size_gb: int = 64):

    chunk_paths = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".pt")
    ])

    if not chunk_paths:
        raise RuntimeError(f"Aucun .pt trouvé dans {input_dir}")

    print(f"📂 {len(chunk_paths)} chunks trouvés")

    map_size = map_size_gb * 1024**3

    env = lmdb.open(
        output_path,
        map_size=map_size,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx_global = 0
    txn = env.begin(write=True)

    try:
        for chunk_path in tqdm(chunk_paths, desc="🔄 Conversion chunks → LMDB"):
            L_batch, ab_batch = torch.load(chunk_path, map_location="cpu")
            n = L_batch.shape[0]

            for i in range(n):
                key = f"{idx_global:08d}".encode("ascii")

                # Convertir en bytes SANS COPIE gros volume
                L_bytes, L_shape, L_dtype = tensor_to_bytes(L_batch[i])
                ab_bytes, ab_shape, ab_dtype = tensor_to_bytes(ab_batch[i])

                record = {
                    "L": (L_bytes, L_shape, L_dtype),
                    "ab": (ab_bytes, ab_shape, ab_dtype)
                }

                txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                idx_global += 1

                if idx_global % COMMIT_EVERY == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn.put(b"__len__", pickle.dumps(idx_global, protocol=pickle.HIGHEST_PROTOCOL))
        txn.commit()

    finally:
        env.sync()
        env.close()

    print(f"✅ LMDB créé avec {idx_global} images.")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_LMDB), exist_ok=True)
    build_lmdb(INPUT_DIR, OUTPUT_LMDB, MAP_SIZE_GB)

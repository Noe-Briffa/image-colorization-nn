import os
import io
import random
import torch
import time
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from torchvision import transforms
import kornia

# ================================
# 🔧 PARAMÈTRES
# ================================
PARQUET_FILES = [
    "validation-00000-of-00002.parquet",
    "validation-00001-of-00002.parquet"
]
OUTPUT_DIR = "dataset/dataset_lab_imagenet256"
TEST_DIR = "dataset/dataset_lab_imagenet256_test"
CHUNK_SIZE_MB = 120
TEST_COUNT = 1000
IMG_SIZE = (256, 256)
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

to_tensor = transforms.ToTensor()
resize = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)


# ================================
# 🧠 CONVERSION RGB → LAB
# ================================
def rgb_to_lab_tensor(image_pil: Image.Image):
    """Convertit une image PIL RGB en tensors L et ab normalisés [0,1]"""
    img = resize(image_pil)
    img_tensor_rgb = to_tensor(img).unsqueeze(0)  # [1,3,H,W]
    img_lab = kornia.color.rgb_to_lab(img_tensor_rgb)

    L = img_lab[:, 0:1, :, :] / 100.0
    ab = (img_lab[:, 1:3, :, :] + 128.0) / 255.0
    return L.squeeze(0), ab.squeeze(0)  # [1,H,W] et [2,H,W]


# ================================
# 💾 SAUVEGARDE ROBUSTE
# ================================
def safe_torch_save(obj, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())

    # Attente + retry pour contourner le verrou Windows
    for _ in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(0.2)  # attend 200 ms et retente
    print(f"⚠️ Impossible de remplacer {path}, fichier temporaire conservé.")


def save_chunk(L_list, ab_list, chunk_idx):
    if not L_list:
        return
    L_batch = torch.stack(L_list)
    ab_batch = torch.stack(ab_list)
    out_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:03d}.pt")
    safe_torch_save((L_batch, ab_batch), out_path)
    print(f"✅ Sauvegardé {out_path} ({len(L_list)} images)")
    L_list.clear()
    ab_list.clear()


def save_test_set(L_list, ab_list):
    if not L_list:
        return
    L_batch = torch.stack(L_list)
    ab_batch = torch.stack(ab_list)
    out_path = os.path.join(TEST_DIR, "test_images.pt")
    safe_torch_save((L_batch, ab_batch), out_path)
    print(f"🎯 Jeu de test sauvegardé ({len(L_list)} images)")
    L_list.clear()
    ab_list.clear()


# ================================
# 🚀 TRAITEMENT PRINCIPAL
# ================================
print("📊 Comptage des images...")
total_images = sum(len(Dataset.from_parquet(p)) for p in PARQUET_FILES)
print(f"Total : {total_images} images")

# Sélection aléatoire de 1000 indices pour le jeu de test
random.seed(42)
test_indices_global = set(random.sample(range(total_images), TEST_COUNT))
print(f"🎯 {len(test_indices_global)} images assignées au test set")

chunk_idx = 0
current_L, current_ab = [], []
test_L, test_ab = [], []
current_size = 0
global_idx = 0

for parquet_file in PARQUET_FILES:
    print(f"\n📦 Lecture de {parquet_file}...")
    dataset = Dataset.from_parquet(parquet_file)

    for row in tqdm(dataset, desc=f"Conversion {parquet_file}"):
        img_data = row["image"]

        try:
            if isinstance(img_data, dict) and "bytes" in img_data:
                image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, Image.Image):
                image = img_data.convert("RGB")
            else:
                global_idx += 1
                continue
        except Exception:
            global_idx += 1
            continue

        # --- Conversion RGB → Lab ---
        L, ab = rgb_to_lab_tensor(image)

        if global_idx in test_indices_global:
            test_L.append(L)
            test_ab.append(ab)
        else:
            current_L.append(L)
            current_ab.append(ab)
            current_size += L.element_size() * L.nelement() + ab.element_size() * ab.nelement()

            # Sauvegarde chunk si taille atteinte
            if current_size >= CHUNK_SIZE_MB * 1024 * 1024:
                save_chunk(current_L, current_ab, chunk_idx)
                chunk_idx += 1
                current_size = 0

        global_idx += 1

# Sauvegarde finale
if current_L:
    save_chunk(current_L, current_ab, chunk_idx)
if test_L:
    save_test_set(test_L, test_ab)

print("\n🎉 Conversion Lab terminée !")
print(f"🧩 Chunks d'entraînement dans : {OUTPUT_DIR}/")
print(f"🧪 Jeu de test dans : {TEST_DIR}/test_images.pt")

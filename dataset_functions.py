import os

import torch
import torchvision.transforms.functional as F
from torch.utils.data import TensorDataset
from PIL import Image
import kornia
from pathlib import Path


def rgb_to_lab(img_path, size):
    # Charger une image
    img = Image.open(img_path).convert("RGB")
    # Redimensionnement
    img = img.resize(size, Image.Resampling.BICUBIC)
    # Convertir en Tensor [0,1]
    img_tensor_rgb = F.to_tensor(img)
    # Passage en batch
    img_tensor_rgb = img_tensor_rgb.unsqueeze(0)
    # Convertir en Lab
    img_tensor_lab = kornia.color.rgb_to_lab(img_tensor_rgb)

    # Normalisation 0-1
    L = img_tensor_lab[:, 0:1, :, :] / 100
    ab = (img_tensor_lab[:, 1:3, :, :] + 128) / 255

    return L, ab


def create_dataset(folder_path, dataset_folder, dataset_name, chunk_size_mb, image_size):
    L_list, ab_list = [], []
    extensions = (".png", ".jpg", ".jpeg")
    img_paths = [str(p) for p in Path(folder_path).rglob("*") if p.suffix.lower() in extensions]

    actuel, total = 1, len(img_paths)
    current_chunk_size = 0
    chunk_index = 1

    for i, img_path in enumerate(img_paths, start=1):
        L, ab = rgb_to_lab(img_path, size=image_size)
        L_list.append(L)
        ab_list.append(ab)
        current_chunk_size += L.element_size() * L.nelement() + ab.element_size() * ab.nelement()

        # sauvegarde par chunk
        if current_chunk_size >= chunk_size_mb * 1024 * 1024 or i == total:
            L_batch = torch.cat(L_list, dim=0)
            ab_batch = torch.cat(ab_list, dim=0)

            torch.save((L_batch, ab_batch), f"{dataset_folder}/{dataset_name}_{chunk_index}.pt")
            print(f"Chunk {chunk_index} sauvegardé ({i}/{total})")
            chunk_index += 1

            L_list.clear()
            ab_list.clear()
            current_chunk_size = 0

        # affichage
        if 100*i/total >= actuel:
            print(f'dataset: {actuel}%')
            actuel += 1



def load_all_chunks(folder_path):
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')])
    datasets = []

    for f in files:
        print(f"Chargement de {f}...")
        L, ab = torch.load(f)
        datasets.append(TensorDataset(L, ab))

    # Concatène virtuellement les datasets
    full_dataset = torch.utils.data.ConcatDataset(datasets)
    return full_dataset

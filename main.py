import os
import time

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset_functions import create_dataset, load_all_chunks
from unet_learning import train

if __name__ == "__main__":
    # Variables
    DATA_DIR = 'imagenet-mini_train'
    DATASET_DIR = 'dataset/dataset_lab_imagenet256.lmdb'
    DATASET_DIR_TEST = 'dataset/dataset_lab_imagenet256_test.lmdb'
    DATASET_NAME = 'dataset_lab'
    CHUNK_SIZE_MB = 120
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 130
    LR_G = 2e-4
    LR_D = 1e-4
    LR_G_FT = 1e-5  # fine-tune
    LR_D_FT = 5e-6  # fine-tune
    EPOCH_FT = 101  # epoch fine-tune
    SUBSET_SIZE = -1
    FEATURES = 48
    SAMPLES_DIR = "samples"
    CKPT = "checkpoints/unet_colorization.pt"

    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

    # Création du dataset
    # create_dataset(folder_path=DATA_DIR,
    #                dataset_folder=DATASET_DIR,
    #                dataset_name=DATASET_NAME,
    #                chunk_size_mb=CHUNK_SIZE_MB,
    #                image_size=IMAGE_SIZE)

    # dataset = load_all_chunks(folder_path=DATASET_DIR)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    t0 = time.time()
    print("training started.")

    # Lancement entraînement
    train(
        dataset_dir=DATASET_DIR,
        dataset_dir_test=DATASET_DIR_TEST,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        LR_G=LR_G,
        LR_D=LR_D,
        LR_G_FT=LR_G_FT,
        LR_D_FT=LR_D_FT,
        epoch_ft=EPOCH_FT,
        num_workers=NUM_WORKERS,
        features=FEATURES,
        subset_size=SUBSET_SIZE,
        samples_dir=SAMPLES_DIR,
        ckpt_path=CKPT,
    )

    print(f"training time: {round(time.time() - t0)}s.")

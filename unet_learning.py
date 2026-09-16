# ==========================
# Imports
# ==========================
import os  # gestion des chemins/fichiers

import torch  # base PyTorch (tensors, autograd)
import torch.nn as nn  # couches de NN
import torch.nn.functional as F  # fonctions (relu, etc.)
from torch.utils.data import Dataset, DataLoader, random_split
from torch import amp
import lpips

import kornia
import kornia.color as K
from tqdm import tqdm
import csv
from math import log10
import time

# ==========================
# Perf : optimisation des kernels
# ==========================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')


# ==========================
# Blocs de base U-Net (version Depthwise-Separable)
# ==========================
class DepthwiseSeparableConv(nn.Module):
    """
    Bloc conv depthwise-separable :
    - conv depthwise 3x3 (groupes = in_ch)
    - conv pointwise 1x1
    - InstanceNorm + ReLU

    Objectif : réduire drastiquement le coût FLOPs tout en gardant une bonne capacité.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DoubleConv(nn.Module):
    """
    Deux blocs depthwise-separable + InstanceNorm + ReLU.
    Version “light” du DoubleConv classique.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """
    Bloc d'encodage: MaxPool(2) pour réduire H,W de moitié, puis DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Bloc de décodage: upsample x2 (par transposed conv), concat skip, DoubleConv.
    - in_ch  : nb de canaux du *concat* (skip + up)
    - out_ch : nb de canaux après DoubleConv
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # on traite uniquement la branche "descendante" dans la transposed conv
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # ajustement spatial éventuel
        diffY = x_skip.size(-2) - x.size(-2)
        diffX = x_skip.size(-1) - x.size(-1)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Channel + Spatial Attention)
    Réduit fortement le coût vs Self-Attention tout en gardant le focus contextuel.
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # ---- Channel Attention ----
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid_channel = nn.Sigmoid()

        # ---- Spatial Attention ----
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # --- Channel attention ---
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att = self.sigmoid_channel(channel_att).view(B, C, 1, 1)
        x = x * channel_att  # pondération par canal

        # --- Spatial attention ---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        spatial_att = self.sigmoid_spatial(spatial_att)
        x = x * spatial_att  # pondération spatiale

        return x


class UNet(nn.Module):
    """
    U-Net 4 down / 4 up version depthwise-separable.
    - in_channels  = 1  (L)
    - out_channels = 2  (ab)
    - features     = largeur de base (64 par défaut)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: int = 64):
        super().__init__()
        f = features
        self.attn = CBAM(features * 16)

        # Encoder
        self.inc = DoubleConv(in_channels, f)  # 1   -> f
        self.down1 = Down(f, f * 2)  # f   -> 2f
        self.down2 = Down(f * 2, f * 4)  # 2f  -> 4f
        self.down3 = Down(f * 4, f * 8)  # 4f  -> 8f
        self.down4 = Down(f * 8, f * 16)  # 8f  -> 16f

        # Decoder
        self.up1 = Up(f * 16, f * 8)  # (16f -> 8f) + 8f  -> 8f
        self.up2 = Up(f * 8, f * 4)  # (8f  -> 4f) + 4f  -> 4f
        self.up3 = Up(f * 4, f * 2)  # (4f  -> 2f) + 2f  -> 2f
        self.up4 = Up(f * 2, f)  # (2f  -> f)  + f   -> f

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)  # f -> 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attn(x5)  # attention CBAM sur le bottleneck

        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.tanh(x)
        return x


# ==========================
# Discriminant (GAN)
# ==========================
class Discriminator(nn.Module):
    """
    PatchGAN conditionnel (pix2pix-like).
    Entrée attendue : concaténation [L, RGB] -> 4 canaux, shape [B,4,H,W]
    Sortie : carte de logits [B,1,h,w] (pas de Sigmoid ici).
    """

    def __init__(self, in_channels: int = 4, feature_channels=(64, 128, 256, 512)):
        super().__init__()

        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Première couche : SANS BatchNorm (classique PatchGAN)
        blocks = [conv_block(in_channels, feature_channels[0], stride=2, norm=False)]
        # Couches intermédiaires : avec BatchNorm
        in_c = feature_channels[0]
        for out_c in feature_channels[1:]:
            # stride=2 sauf l’avant-dernière si tu veux une carte plus dense
            blocks.append(conv_block(in_c, out_c, stride=2, norm=True))
            in_c = out_c

        self.body = nn.Sequential(*blocks)
        # Dernière conv stride=1 (réduit moins la carte et garde le côté "patch")
        self.tail = nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1)

        # Optionnel : init "DCGAN-like"
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x = torch.cat([L, rgb], dim=1)  # L:[B,1,H,W], rgb:[B,3,H,W]
        """
        h = self.body(x)
        out = self.tail(h)  # logits (pas de Sigmoid)
        return out


# ==========================
# Utilitaires
# ==========================
@torch.no_grad()
def lab01_to_rgb01(L01: torch.Tensor, ab01: torch.Tensor) -> torch.Tensor:
    """
    Version ultra optimisée Lab(0..1) -> RGB(0..1), full GPU, vectorisée.
    L01 : [B,1,H,W]  in [0,1]
    ab01: [B,2,H,W]  in [0,1]  (ou [-1,1] déjà rescalé avant appel)
    """
    # On remet dans les échelles Lab standard
    L = L01 * 100.0
    a = ab01[:, 0:1] * 255.0 - 128.0
    b = ab01[:, 1:2] * 255.0 - 128.0

    # Lab -> XYZ (D65)
    # Formules standard CIE
    y = (L + 16.0) / 116.0
    x = y + a / 500.0
    z = y - b / 200.0

    eps = 6.0 / 29.0

    def f_inv(t):
        return torch.where(
            t > eps,
            t ** 3,
            3 * (eps ** 2) * (t - 4.0 / 29.0)
        )

    x = f_inv(x)
    y = f_inv(y)
    z = f_inv(z)

    # Référence D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = Xn * x
    Y = Yn * y
    Z = Zn * z

    # XYZ -> RGB linéaire (sRGB)
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    rgb = torch.cat([r, g, b], dim=1)

    # Gamma sRGB
    thresh = 0.0031308
    low = 12.92 * rgb
    high = 1.055 * torch.clamp(rgb, min=0.0) ** (1.0 / 2.4) - 0.055
    rgb = torch.where(rgb <= thresh, low, high)

    return rgb.clamp(0.0, 1.0)


@torch.no_grad()
def lab01_to_rgb01_fast(L01: torch.Tensor, ab01: torch.Tensor) -> torch.Tensor:
    """
    Version ultra rapide Lab→RGB, basée sur kornia (CUDA optimisée).
    Entrées :
        L01 ∈ [0,1], shape [B,1,H,W]
        ab01 ∈ [0,1], shape [B,2,H,W]
    Retour :
        rgb01 ∈ [0,1], shape [B,3,H,W]
    """

    # Convertir Lab normalisé en Lab "vrai"
    # L ∈ [0,100]
    L = L01 * 100

    # a,b ∈ [-128,128]
    ab = ab01 * 255 - 128

    Lab = torch.cat([L, ab], dim=1)  # [B,3,H,W]

    # Kornia lab_to_rgb attend L∈[0,100], ab∈[-128,128]
    rgb = K.lab_to_rgb(Lab)  # retourne [0,1]

    return torch.clamp(rgb, 0.0, 1.0)


@torch.no_grad()
def colorize_L_batch(model: UNet, L01: torch.Tensor) -> torch.Tensor:
    """
    Inférence: prend L normalisé [B,1,H,W], renvoie ab prédits [B,2,H,W] en 0..1.
    """
    model.eval()
    device = next(model.parameters()).device
    L01 = L01.to(device)
    ab01 = model(L01)
    return ab01


def save_grid_rgb(rgb01: torch.Tensor, path: str, nrow: int = 8) -> None:
    """
    Sauvegarde une grille d'images RGB [0..1] au chemin donné.
    """
    # torchvision.utils n'est pas strictement nécessaire: on assemble à la main
    # mais pour la simplicité on reste en kornia/torch (pas d'IO direct ici).
    # On convertit en uint8 et on sauve via PIL.
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    grid = make_grid(rgb01, nrow=nrow)  # [3,H*,W*]
    img = to_pil_image(grid)  # uint8 PIL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# ==========================
# Dataset : Lazy Loading
# ==========================
class LazyLabDatasetLMDB(Dataset):
    """
    Dataset basé sur LMDB (compatible avec les données stockées sous forme de bytes).
    - ne charge jamais plus que nécessaire en RAM
    - reconstruit L et ab à partir des bytes stockés
    """

    def __init__(self, lmdb_path: str):
        import lmdb
        import pickle

        self.lmdb_path = lmdb_path
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            subdir=False,
        )

        with self.env.begin(write=False) as txn:
            len_bytes = txn.get(b"__len__")
            if len_bytes is None:
                raise RuntimeError(f"Clé __len__ absente dans {lmdb_path}")
            self.length = pickle.loads(len_bytes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        import pickle
        import numpy as np
        import torch

        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        key = f"{idx:08d}".encode("ascii")

        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        if value is None:
            raise KeyError(f"Clé {key!r} absente dans {self.lmdb_path}")

        record = pickle.loads(value)

        # Décode L
        L_bytes, L_shape, L_dtype = record["L"]
        L_np = np.frombuffer(L_bytes, dtype=L_dtype).reshape(L_shape)
        L = torch.from_numpy(L_np.copy()).float()

        # Décode ab
        ab_bytes, ab_shape, ab_dtype = record["ab"]
        ab_np = np.frombuffer(ab_bytes, dtype=ab_dtype).reshape(ab_shape)
        ab = torch.from_numpy(ab_np).float()

        return L, ab


# ==========================
# Logging System
# ==========================
class TrainingLogger:
    def __init__(self, csv_path):
        self.batch_metrics = None
        self.csv_path = csv_path
        self.start_time = None
        self.metrics = []
        self.reset_batch()

        # création du CSV avec en-tête
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "batch",
                    "time_data",
                    "time_forward_G", "time_forward_G2", "time_backward_G",
                    "time_forward_D", "time_backward_D",
                    "time_batch_cpu", "time_batch_gpu", "time_batch_total",
                    "loss_G", "loss_G_GAN", "loss_G_L1", "loss_G_PERC",
                    "loss_D", "loss_D_real", "loss_D_fake",
                    "PSNR", "SSIM", "deltaE",
                    "LPIPS_raw", "LPIPS_weight",
                    "gpu_alloc_MB", "gpu_reserved_MB"
                ])

    def reset_batch(self):
        self.batch_metrics = {
            "time_data": 0,
            "time_forward_G": 0,
            "time_forward_G2": 0,
            "time_backward_G": 0,
            "time_forward_D": 0,
            "time_backward_D": 0,
            "time_batch_cpu": 0,
            "time_batch_gpu": 0,
            "time_batch_total": 0,

            "loss_G": 0,
            "loss_G_GAN": 0,
            "loss_G_L1": 0,
            "loss_G_PERC": 0,

            "loss_D": 0,
            "loss_D_real": 0,
            "loss_D_fake": 0,

            "PSNR": 0,
            "SSIM": 0,
            "deltaE": 0,

            "LPIPS_raw": 0,
            "LPIPS_weight": 0,

            "gpu_alloc_MB": 0,
            "gpu_reserved_MB": 0
        }

    def update(self, key, value):
        self.batch_metrics[key] = value

    def record(self, epoch, batch):
        row = [epoch, batch]
        row += [self.batch_metrics[k] for k in [
            "time_data",
            "time_forward_G", "time_forward_G2", "time_backward_G",
            "time_forward_D", "time_backward_D",
            "time_batch_cpu", "time_batch_gpu", "time_batch_total",
            "loss_G", "loss_G_GAN", "loss_G_L1", "loss_G_PERC",
            "loss_D", "loss_D_real", "loss_D_fake",
            "PSNR", "SSIM", "deltaE",
            "LPIPS_raw", "LPIPS_weight",
            "gpu_alloc_MB", "gpu_reserved_MB"
        ]]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def compute_metrics(self, rgb_fake, rgb_real):
        # PSNR
        mse = torch.mean((rgb_fake - rgb_real) ** 2)
        psnr = 10 * log10(1 / mse.item()) if mse.item() > 0 else 0

        # SSIM
        ssim = kornia.metrics.ssim(rgb_fake, rgb_real, 11).mean().item()

        # ΔE (erreur de couleur)
        Lab_fake = kornia.color.rgb_to_lab(rgb_fake)
        Lab_real = kornia.color.rgb_to_lab(rgb_real)
        deltaE = torch.mean(torch.linalg.norm(Lab_fake - Lab_real, dim=1)).item()

        return psnr, ssim, deltaE


# ==========================
# Entraînement
# ==========================
def train(
        dataset_dir: str,
        dataset_dir_test: str,
        batch_size: int = 32,
        epochs: int = 20,
        LR_G: float = 2e-4,
        LR_D: float = 1e-4,
        LR_G_FT: float = 1e-5,
        LR_D_FT: float = 5e-6,
        epoch_ft: int = 100,
        num_workers: int = 4,
        features: int = 64,
        subset_size: int = -1,
        samples_dir: str = "samples",
        ckpt_path: str = "checkpoints/unet_colorization.pt"
):
    """
    Lance l'entraînement du U-Net:
    - dataset_dir : dossier contenant les fichiers .pt (chunks L,ab)
    - dataset_dir : dossier contenant les fichiers .pt de test (chunks L,ab)
    - batch_size  : taille des batchs
    - epochs      : nombre d'époques d'entraînement
    - lr          : learning rate Adam
    - num_workers : workers DataLoader
    - features    : largeur de base du U-Net (64 recommandé pour CIFAR)
    - samples_dir : où sauver les aperçus RGB
    - ckpt_path   : où sauver le modèle
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print("🧩 Chargement du dataset en mode Lazy (un chunk à la fois)...")
    dataset = LazyLabDatasetLMDB(dataset_dir)
    dataset_test = LazyLabDatasetLMDB(dataset_dir_test)

    if subset_size != -1: dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # Lazy loading -> un seul worker pour éviter les collisions de torch.load
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None
    )

    L_fixed, ab_fixed = next(iter(DataLoader(dataset_test, batch_size=32, shuffle=True, num_workers=0)))
    L_fixed, ab_fixed = L_fixed.to(device), ab_fixed.to(device)

    # === Modèles ===
    G = UNet(in_channels=1, out_channels=2, features=features).to(
        device=device,
        memory_format=torch.channels_last
    )
    D = Discriminator(in_channels=3 + 1).to(
        device=device,
        memory_format=torch.channels_last
    )  # conditionnel : L(1) + RGB(3)

    # === LPIPS perceptual loss ===
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    lpips_loss.eval()  # on ne l'entraîne pas
    for p in lpips_loss.parameters():
        p.requires_grad = False

    # === Reprise checkpoint si dispo ===
    start_epoch = 1
    ckpt_dir = os.path.dirname(ckpt_path)
    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    existing_ckpts = [f for f in os.listdir(ckpt_dir) if
                      f.startswith(base_name.replace(".pt", "")) and f.endswith(".pt")]
    if existing_ckpts:
        # récupérer le plus grand numéro d'epoch
        latest = sorted(existing_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        ckpt_full_path = os.path.join(ckpt_dir, latest)
        print(f"🔄 Reprise à partir du checkpoint {ckpt_full_path}")
        checkpoint = torch.load(ckpt_full_path, map_location=device)

        # Chargement tolérant pour compatibilité ancienne version
        missing, unexpected = G.load_state_dict(checkpoint["G"], strict=False)
        print(f"ℹ️ G chargé avec {len(missing)} clés manquantes et {len(unexpected)} inattendues.")
        D.load_state_dict(checkpoint["D"], strict=False)

        start_epoch = checkpoint.get("epoch", 0) + 1
    else:
        print("🚀 Aucun checkpoint trouvé, nouvel entraînement.")

    G.eval()
    with torch.no_grad():
        rgb_ref = lab01_to_rgb01_fast(L_fixed.cpu(), ab_fixed.cpu())
        save_grid_rgb(
            rgb_ref,
            os.path.join(samples_dir, "epoch0_reference_lab.png"),
            nrow=8
        )
    print("✅ Images de référence sauvegardées (L + ab réels)")
    G.train()

    # === Fonctions de perte et optim ===
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    scaler_G = amp.GradScaler('cuda')
    scaler_D = amp.GradScaler('cuda')

    logger = TrainingLogger("checkpoints/full_metrics.csv")

    # warm-up GPU
    with torch.no_grad():
        dummy_L = torch.randn(1, 1, 256, 256, device=device).to(memory_format=torch.channels_last)
        dummy_ab = torch.rand(1, 2, 256, 256, device=device) * 2 - 1  # proche de la sortie G
        _ = G(dummy_L)
        rgb_dummy = lab01_to_rgb01_fast(dummy_L, (dummy_ab + 1) / 2)
        _ = D(torch.cat([dummy_L, rgb_dummy], dim=1))
    torch.cuda.synchronize()

    print("🚀 Début de l’entraînement CGAN pour colorisation…")

    # === Boucle d'entraînement ===
    for epoch in range(start_epoch, epochs + 1):
        if epoch == epoch_ft:
            opt_G = torch.optim.Adam(G.parameters(), lr=LR_G_FT, betas=(0.5, 0.999))
            opt_D = torch.optim.Adam(D.parameters(), lr=LR_D_FT, betas=(0.5, 0.999))
        G.train()
        D.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_idx, (L, ab) in enumerate(pbar):
            # mesure CPU globale (inclut DataLoader + transferts)
            t_cpu_start = time.time()

            t_data_start = time.time()
            L = L.to(device, non_blocking=True, memory_format=torch.channels_last)
            ab = ab.to(device, non_blocking=True, memory_format=torch.channels_last)
            t_data = time.time() - t_data_start

            # Mesure GPU avant
            gpu_alloc = torch.cuda.memory_allocated() / 1e6
            gpu_reserved = torch.cuda.memory_reserved() / 1e6

            # === 1️⃣ Génération ===
            t_fG_start = time.time()
            with amp.autocast(device_type='cuda'):
                ab_fake = G(L)  # tanh → [-1,1]

            t_fG = time.time() - t_fG_start

            # === 2️⃣ Entraînement du Discriminateur ===
            t_fD_start = time.time()

            with torch.no_grad():
                # Lab(0..1) -> RGB(0..1) en pleine résolution pour D

                rgb_real_256 = lab01_to_rgb01_fast(L, ab)  # [B,3,256,256]
                rgb_fake_256_ng = lab01_to_rgb01_fast(L, (ab_fake + 1) / 2)  # [B,3,256,256]

            rgb_fake_256_for_D = rgb_fake_256_ng.detach()

            with amp.autocast(device_type='cuda'):
                D_real = D(torch.cat([L, rgb_real_256], dim=1))
                D_fake = D(torch.cat([L, rgb_fake_256_for_D.detach()], dim=1))

                loss_D_real = criterion_GAN(D_real, torch.ones_like(D_real) * 0.9)
                loss_D_fake = criterion_GAN(D_fake, torch.zeros_like(D_fake) + 0.1)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
            t_fD = time.time() - t_fD_start

            opt_D.zero_grad(set_to_none=True)
            t_bD_start = time.time()
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            t_bD = time.time() - t_bD_start

            # === 3️⃣ Entraînement du Générateur ===
            t_fG2_start = time.time()
            with torch.no_grad():
                # Downsample Lab en 128
                L_128 = F.interpolate(L, size=(128, 128), mode="bilinear", align_corners=False)
                ab_fake_128 = F.interpolate((ab_fake + 1) / 2, size=(128, 128),
                                            mode="bilinear", align_corners=False)
                ab_real_128 = F.interpolate(ab, size=(128, 128),
                                            mode="bilinear", align_corners=False)

                # Lab(0..1) -> RGB(0..1) en 128×128
                rgb_fake_128_ng = lab01_to_rgb01_fast(L_128, ab_fake_128)
                rgb_real_128_ng = lab01_to_rgb01_fast(L_128, ab_real_128)

            # Bridge de gradient 128×128 (STE pour LPIPS)
            grad_bridge_128 = ab_fake_128[:, :1, :, :].expand_as(rgb_fake_128_ng)
            rgb_fake_128 = rgb_fake_128_ng + (grad_bridge_128 - grad_bridge_128.detach()) * 0
            rgb_real_128 = rgb_real_128_ng

            # -------- 3.2 RGB 256 pour GAN (avec STE léger) --------
            # On peut aussi faire un petit bridge 256 pour que le GAN voie rgb comme fonction de ab_fake
            grad_bridge_256 = ab_fake[:, :1, :, :].expand_as(rgb_fake_256_ng)
            rgb_fake_256 = rgb_fake_256_ng + (grad_bridge_256 - grad_bridge_256.detach()) * 0

            with amp.autocast(device_type="cuda"):
                # --- GAN : le Générateur veut tromper D en 256×256 ---
                D_fake_for_G = D(torch.cat([L, rgb_fake_256], dim=1))
                loss_G_GAN = criterion_GAN(D_fake_for_G, torch.ones_like(D_fake_for_G))

                # --- L1 sur l'espace ab ---
                loss_G_L1 = criterion_L1((ab_fake + 1) / 2, ab)

                # --- LPIPS : perceptuel sur 128×128, sous-batch de 8 images ---
                loss_lpips = torch.tensor(0.0, device=device)
                loss_G_PERC = torch.tensor(0.0, device=device)
                lpips_w = 0.0

                if epoch >= epoch_ft:
                    if epoch <= epoch_ft + 20:
                        lpips_w = 10.0 * ((epoch - epoch_ft) / 20.0)
                    else:
                        lpips_w = 10.0

                    b_lp = min(rgb_fake_128.size(0), 8)
                    rgb_fake_lp = rgb_fake_128[:b_lp] * 2.0 - 1.0
                    rgb_real_lp = rgb_real_128[:b_lp] * 2.0 - 1.0

                    loss_lpips = lpips_loss(rgb_fake_lp, rgb_real_lp).mean()
                    loss_G_PERC = loss_lpips

                    loss_G = loss_G_GAN + 15.0 * loss_G_L1 + lpips_w * loss_lpips
                else:
                    loss_G = loss_G_GAN + 30.0 * loss_G_L1

            t_fG2 = time.time() - t_fG2_start

            opt_G.zero_grad(set_to_none=True)
            t_bG_start = time.time()
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            t_bG = time.time() - t_bG_start

            # === METRICS ===
            with torch.no_grad():
                psnr, ssim, deltaE = logger.compute_metrics(rgb_fake_128, rgb_real_128)

            # temps GPU = somme des parties principales (approx propre)
            time_batch_gpu = t_fG + t_fG2 + t_bG + t_fD + t_bD

            # temps CPU global (de l'entrée dans la boucle jusqu'ici)
            t_cpu = time.time() - t_cpu_start

            # temps total = CPU global (qui inclut tout)
            t_total = t_cpu

            # === LOGGING ===
            logger.update("time_data", t_data)
            logger.update("time_forward_G", t_fG)
            logger.update("time_forward_G2", t_fG2)
            logger.update("time_backward_G", t_bG)
            logger.update("time_forward_D", t_fD)
            logger.update("time_backward_D", t_bD)

            logger.update("loss_G", loss_G.item())
            logger.update("loss_G_GAN", loss_G_GAN.item())
            logger.update("loss_G_L1", loss_G_L1.item())
            logger.update("loss_G_PERC", loss_G_PERC.item() if epoch >= epoch_ft else 0.0)

            logger.update("loss_D", loss_D.item())
            logger.update("loss_D_real", loss_D_real.item())
            logger.update("loss_D_fake", loss_D_fake.item())

            logger.update("LPIPS_raw", loss_lpips.item())
            logger.update("LPIPS_weight", lpips_w)

            logger.update("PSNR", psnr)
            logger.update("SSIM", ssim)
            logger.update("deltaE", deltaE)

            logger.update("gpu_alloc_MB", gpu_alloc)
            logger.update("gpu_reserved_MB", gpu_reserved)

            logger.update("time_batch_cpu", t_cpu)
            logger.update("time_batch_gpu", time_batch_gpu)
            logger.update("time_batch_total", t_total)

            logger.record(epoch, batch_idx)

        print(f"[Epoch {epoch}/{epochs}] is over.")

        # === 4️⃣ Sauvegarde visuelle ===
        G.eval()
        with torch.no_grad():
            ab_pred = G(L_fixed)

            # color boosting
            s = 1.2
            ab_boost = ((ab_pred + 1) / 2 - 0.5) * s + 0.5
            rgb_pred = lab01_to_rgb01_fast(L_fixed, ab_boost.clamp(0, 1))

            save_grid_rgb(rgb_pred, os.path.join(samples_dir, f"epoch{epoch:03d}.png"), nrow=8)

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "epoch": epoch
        }, ckpt_path.replace(".pt", f"_{epoch}.pt"))

    print("✅ Entraînement terminé avec succès !")

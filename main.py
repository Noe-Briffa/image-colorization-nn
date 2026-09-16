import argparse
import time
from pathlib import Path

from unet_learning import train


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: Path) -> Path:
    """Interprète les chemins relatifs depuis la racine du dépôt."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de colorisation U-Net/cGAN.")
    parser.add_argument("--dataset", type=Path, required=True, help="LMDB d'entraînement.")
    parser.add_argument("--test-dataset", type=Path, required=True, help="LMDB tenu à part.")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--subset-size", type=int, default=-1)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--lr-g-finetune", type=float, default=1e-5)
    parser.add_argument("--lr-d-finetune", type=float, default=5e-6)
    parser.add_argument("--finetune-epoch", type=int, default=101)
    parser.add_argument(
        "--checkpoint-base",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_colorization.pt",
    )
    parser.add_argument("--resume", type=Path, help="Checkpoint précis à reprendre.")
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "samples")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "full_metrics.csv",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (("dataset", args.dataset), ("test-dataset", args.test_dataset)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} LMDB introuvable: {path}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.features <= 0:
        raise ValueError("epochs, batch-size et features doivent être supérieurs à zéro.")


def main() -> None:
    args = parse_args()
    for name in ("dataset", "test_dataset", "checkpoint_base", "samples_dir", "metrics", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve_project_path(value))
    validate_args(args)
    started_at = time.time()
    train(
        dataset_dir=str(args.dataset),
        dataset_dir_test=str(args.test_dataset),
        batch_size=args.batch_size,
        epochs=args.epochs,
        LR_G=args.lr_g,
        LR_D=args.lr_d,
        LR_G_FT=args.lr_g_finetune,
        LR_D_FT=args.lr_d_finetune,
        epoch_ft=args.finetune_epoch,
        num_workers=0,
        features=args.features,
        subset_size=args.subset_size,
        samples_dir=str(args.samples_dir),
        ckpt_path=str(args.checkpoint_base),
        metrics_path=str(args.metrics),
        device_name=args.device,
        seed=args.seed,
        resume_path=str(args.resume) if args.resume else None,
    )
    print(f"Entraînement terminé en {round(time.time() - started_at)} s.")


if __name__ == "__main__":
    main()

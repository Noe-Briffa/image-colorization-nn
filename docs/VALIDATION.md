# Preuve de validation V1

Exécution locale du 16 septembre 2026, sur une RTX 3060 Laptop 6 Go :

- Python 3.11.16
- PyTorch 2.7.0+cu128, CUDA disponible (12.8)
- Kornia 0.8.1, LPIPS 0.1.4, Gradio 5.49.1, LMDB 1.7.5
- Checkpoint historique chargé strictement : `unet_colorization_119.pt`
- Tests unitaires : `4 passed`
- Inférence : 12/12 images PNG produites, dimensions et mode RGB conservés

## Évaluation indépendante

Commande :

```powershell
python evaluate.py `
  --dataset dataset\dataset_lab_imagenet256_test.lmdb `
  --checkpoint checkpoints\unet_colorization_119.pt `
  --max-samples 100 --seed 42 `
  --output artifacts\evaluation_v1 --device cuda --features 48
```

Résultats V1 (100 indices fixes, seed 42) :

| métrique | moyenne | écart-type | min | max |
|---|---:|---:|---:|---:|
| PSNR (dB) | 23.2869 | 4.6141 | 10.3565 | 36.2112 |
| SSIM | 0.9402 | 0.0505 | 0.7552 | 0.9947 |
| DeltaE | 14.8435 | 8.0088 | 2.9878 | 45.6908 |
| LPIPS (VGG) | 0.1938 | 0.0811 | 0.0145 | 0.4270 |

Une seconde exécution dans `artifacts/evaluation_repeat/` a sélectionné les mêmes
indices et produit les mêmes valeurs dans cet environnement CUDA. Les résultats
restent indicatifs : ils évaluent un échantillon ImageNet tenu à part, pas une
restitution historique garantie.

Les six triptyques générés sont conservés localement dans `artifacts/` (dossier
ignoré par Git). Ne les publier qu'après vérification des droits de diffusion.

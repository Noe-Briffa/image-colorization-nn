# Colorisation d'images par réseau de neurones

Projet personnel de colorisation d'images noir et blanc. Un U-Net prédit les canaux chromatiques `a,b` dans l'espace Lab à partir du canal de luminance `L`.

## Ce que fait le projet

- Prépare des images RGB sous forme de paires `L` / `ab`.
- Entraîne un générateur U-Net avec attention CBAM.
- Explore une perte adversariale via un discriminateur PatchGAN et une perte perceptuelle LPIPS.
- Colorise des images depuis un dossier local.

La colorisation est une prédiction plausible, non une restauration historique fiable. Les couleurs d'origine ne peuvent pas être déduites avec certitude depuis une image noir et blanc.

## Architecture

`image RGB -> Lab -> L -> U-Net + CBAM -> ab -> Lab -> image colorisée`

Le générateur emploie des convolutions depthwise-separable. Les données sont lues au format LMDB pour éviter de charger l'ensemble du jeu de données en mémoire.

## Installation

Python 3.10 ou 3.11 recommandé. Créer un environnement virtuel, installer une version de PyTorch compatible avec la carte graphique, puis :

```bash
pip install -r requirements.txt
```

## Inférence

Placer des images dans un dossier, puis lancer :

```bash
python YT/inference_colorize.py --input chemin/vers/images --output resultats --checkpoint checkpoints/unet_colorization_119.pt
```

Le checkpoint n'est pas inclus dans le dépôt. Il doit être téléchargé séparément depuis une release ou fourni localement.

## Évaluation

Les métriques historiques présentes localement ont été calculées pendant entraînement. Elles ne constituent pas une évaluation indépendante. Une prochaine version fournira un protocole de validation séparé.

## Contenu volontairement exclu du dépôt

Jeux de données, checkpoints intermédiaires, sorties générées, vidéos et environnement local. Cela évite un dépôt de plusieurs dizaines de gigaoctets et respecte les licences des sources de données.

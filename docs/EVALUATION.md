# Protocole d'évaluation V1

- Source : LMDB test distinct du LMDB d'entraînement.
- Échantillon : 100 indices sans remise, sélectionnés avec `random.Random(42)` puis triés.
- Modèle : `unet_colorization_119.pt`, `features=48`, mode évaluation.
- Mesures : PSNR et SSIM en RGB `[0,1]`, DeltaE euclidien dans Lab, LPIPS VGG en RGB `[-1,1]`.
- Sorties : `metrics.json`, liste complète des indices et six triptyques entrée/prédiction/référence.
- Reproductibilité : mêmes versions, checkpoint, LMDB, seed et device. De légères différences numériques CPU/GPU restent possibles.

Les résultats ne doivent être publiés qu'après exécution réussie du protocole. Les métriques de batch historiques ne sont pas des métriques de validation.

# Projet_Deep_Learning

Papier originel: https://arxiv.org/abs/1611.07004.pdf

## Description du dataset

* Nous utilisons le dataset du papier original (pix2pix) : [Télécharger le dataset](https://drive.google.com/file/d/1s5a2UeJR4H_KJ-nV4NmRMkBHr3zn20Tf/view?usp=sharing)
* Un exemple est disponible sur ce repo. 
* Une fois téléchargé, il est nécessaire de remplacer le dossier '/maps' ici présent par celui téléchargé.

## Hyper paramètres

* Taille de batch: **1**
* Taille des images: **256 x 256**
* Taux d'apprentissage: **0.0002**
* Moments: [β1, β2] = **[0.5, 0.999]**
* λ_L1 = **100**

## Architecture du générateur

* Architecture: **U-Net 256**
* Entrée 3 x 256 x 256 - Encoder C64 - C128 - C256 - C512 - C512 - C512
* C512 (Espace latent)
* Decoder DC1024 - DC1024 - DC1024 - DC512 - DC256 - DC128 - Sortie 3 x 256 x 256
* Reference: https://arxiv.org/pdf/1505.04597.pdf

## Architecture du discriminateur

* Architecture: **PatchGan** discriminateur avec champs d'activation de chaque patch tel que rf = 70x70
* C64 (no norm, 4,2,1)-C128 (4,2,1)-C256 (4,2,1)-C512 (4,1,1) - Channel 1 (4,1,1), où C correspond à Convolution - BatchNorm - LeakyReLU
* Activations avec LeakyReLU(0.2)
* BatchNorm sur chaque couche excepté C64
* Les deux dernières couches ont: kernel_size=4, stride=1, padding=1

## Visualisation pendant l'entrainement

* Nous utilisons TensorBoard pour visualiser les loss et les images à chaque époque.
* Vous trouverez la documentation [ici](https://pytorch.org/docs/stable/tensorboard.html)
* Vous pouvez installer et lancer TensorBoard avec:
``
pip install tensorboard
tensorboard --logdir=runs
``
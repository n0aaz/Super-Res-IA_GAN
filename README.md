# Super-Res-IA_GAN

## Introduction
Algorithme de super résolution par IA - SRGAN
Le but de ce programme est d'implémenter l'architecture SRGAN telle que décrite dans l'article _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_ (arXiv:1609.04802v5) Puis d'y apporter notre petite touche personnelle en l'entraînant et développant par dessus.

## Dépendances : 
Rappel : utiliser `pip install xxx` pour installer le module xxx sur python
- Pillow
- Tensorflow (préférablement avec support GPU/Cuda)
- Keras
- glob3
- numpy
- matplotlib
- datetime
- os

Ligne à copier/coller pour les fainéants : 
```pip install pillow tensorflow keras glob numpy matplotlib datetime os```

## Fichiers, structure et explications:

Fichiers : 
- _DataLoader.py_ : contient tout le préprocessing des données d'entraînement, récupère les images dans une base de données qui se situe dans _datasets_, normalise les images, les redimensionne à une taille prédéterminée et effectue du data augmenting pour démultiplier la quantité du training set
- _Discriminator.py_ : la classe qui va nous générer le modèle Keras du discriminateur . Il s'agit du modèle qui s'occupe de discriminer c'est à dire de déterminer pour l'entraînement si une image qui lui est fournie est une image générée par le générateur ou bien une vraie image de la base de données.
- _Generator.py_ : la classe qui va générer le modèle du générateur. Il s'agit du modèle qui s'occupe de générer l'image haute résolution à partir d'une image basse résolution
- _Vgg.py_: idem que pour les précédents , contient le modèle et l'architecture du réseau pré-entraîné VGG19.
- _ImageSuperRes.py_ : tout ce qui concerne le traitement des grandes images. Ici on s'occupe de découper une image en petites images de taille donnée (par défaut 112*112) , d'appliquer la superrésoluton en batch sur chacune de ces petites images, puis à reconstituer les versions agrandies de ces petits blocs en une seule grande image. Inclut également un ajout de pixels noirs lorsque l'image d'origine n'est pas de taille parfaitement multiple de 112 ainsi que d'une suppression (tant que possible) des artéfacts introduits par la super résolution.
- _README.md_ : ce que vous êtes présentement en train de lire
- _SRGAN.py_ : le coeur du projet, génère et compile les modèles du générateur, discriminateur et VGG pour créer le modèle combiné. Gère également tout l'entraînement du modèle de base puis sa sauvegarde/son chargement depuis le dossier _Model/_
- _Script_Nocturne.py_ : script python utilisé pour pouvoir être lancé pendant la nuit, à distance, sans avoir besoin d'avoir ma propre machine d'allumée. Permet de lancer un training sur un temps donné en sauvegardant périodiquement le modèle ainsi qu'en enregistrant un aperçu de l'entraînement dans _/images/MS_Coco/training/_ . Utilisé en lançant depuis le terminal SSH la commande suivante: ```nohup python Script_Nocturne.py > logNocturne.txt```
- _logNocturne.txt_ : fichier texte qui sert à récupérer la sortie de terminal de notre ```Script_Nocturne.py```, puisque le script tourne au travers de Nohup (pour no hangup) lorsqu'aucun PC n'est connecté à la machine de calcul, sans cela la sortie et les erreurs du terminal iraient dans le vide.

## Utilisation : 


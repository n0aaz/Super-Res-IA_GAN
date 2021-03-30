# Super-Res-IA_GAN

## Introduction
Algorithme de super résolution par IA - SRGAN
Le but de ce programme est d'implémenter l'architecture SRGAN telle que décrite dans l'article _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_ (arXiv:1609.04802v5) Puis d'y apporter notre petite touche personnelle en l'entraînant et développant par dessus.

## Dépendances : 
Rappel : utiliser `pip install xxx` pour installer le module xxx sur python
- Pillow
- Tensorflow (préférablement avec support GPU/Cuda)
- Keras
- glob
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

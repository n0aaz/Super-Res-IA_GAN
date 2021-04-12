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
```bash
pip install pillow tensorflow keras glob numpy matplotlib datetime os 
```

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

***

## Utilisation : 

### Appeler le modèle combiné : 

Toute la magie opère dans le fichier _SRGAN.py_ que nous avons programmé comme un objet python contenant tout ce dont on a besoin pour s'en servir.
Ainsi pour pouvoir l'entraîner il faut tout simplement appeler la classe SRGAN comme suit : 
```Python
from SRGAN import SRGAN

# initialiser l'objet SRGAN: 

SR= SRGAN()

# Quelques flags et paramètres utiles pour l'entraînement:
SR.train_batch_size = 4 # taille du batch size, dépend de votre quantité de VRAM , si out of memory, diminuer cette valeur
SR.auto_sauvegarde= True # Permet de sauvegarder le modèle automatiquement tous les N epoch
SR.intervalle_sauvegarde=50 # Le fameux N dont je vous parlais.

```

### Entraîner le modèle : 

Vous pouvez choisir de fonctionnner de deux manières au vu de l'implémentation actuelle : 
1. Entraîner le modèle "nu", c'est à dire à partir de zéro
2. Charger l'état du modèle précédemment sauvegardé (merci l'auto sauvegarde) et reprendre l'entraînement à partir de là 

#### 1 : Entrainer le modèle nu
Une seule ligne, une seule fonction, pas besoin de vous embêter ! 
```Python
SR.train(10000) #Pour lancer un entraînement sur 10000 epoch ! Attention c'est long
```

#### 2 : Charger puis entraîner le modèle
Un peu moins facile cette fois mais je suis sûr que vous ne devriez pas avoir de soucis, après tout vous êtes le genre de personne qui lit la documentation n'est ce pas?
```Python
SR.charger_modeles()
SR.train(10000)
```

### Une fois le modèle entraîné : générer des images haute résolution! 

Une fois le modèle entraîné, si vous voulez générer des images vous n'êtes pas obligé de charger le modèle complet (extrêmement lourd, 700MB rien que pour le Discriminateur). En effet un avantage majeur du modèle GAN est qu'il permet d'avoir un générateur de taille ridiculement petite une fois entraîné, ici c'est 1.4MB soit un peu plus que l'équivalent d'une bonne vieille disquette.

```Python
SR.charger_generateur()

# Aller chercher une image avec une librairie qui vous arrange
from PIL import Image
image_lowres= np.asarray(Image.open('chemin/vers/votre/image.jpg'))
# Attention: cette image doit toujours être de taille 112*112*3 ! 
image_lowres_normalisee= ((image_lowres/255.0)-0.5)*2.0 
# toujours avoir en entrée des valeurs de pixel entre -1 et 1!

# La prédiction de l'image haute résolution
image_highres = SR.generateur.predict(image_lowres_normalisee)
```

**ATTENTION: Si vous passez par cette méthode vous effectuez une prédiction directement en passant dans le réseau de neurones, c'est pourquoi il faut absolument que l'image d'entrée soit de taille 112x112x3 et que chaque pixel ait des valeurs comprises dans [-1;1]**

#### L'alternative préférée et préférable: la classe ImageSuperRes

Cette manière de prédire des images n'est pas la plus pratique assurément c'est pourquoi nous avons créé un wrapper autour de ce travail sur des petites images qui s'occupe de tout automatiser. Voici la classe ImageSuperRes : 

**Initialisation et appel:**

```Python
from ImageSuperRes import ImageSuperRes

ISR= ImageSuperRes()
imageHauteResolution = ISR.superResolution('Le/chemin/vers/votre/image.png')
```
Cette fois ci , la seule limite de taille de votre image "basse résolution" est la mémoire disponible sur votre GPU, cette fonction découpe votre image en petites images et effectue la super résolution sur des bandes de taille largeur*112, si ces bandes sont trop grandes, vous allez avoir un Out Of Memory (OOM)

![image](https://user-images.githubusercontent.com/7826562/113746102-3a618680-9706-11eb-9bc8-d1dff761e6b7.png)

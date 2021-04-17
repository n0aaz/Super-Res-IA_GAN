Ce dossier est volontairement vide : le dataset qui a été utilisé pour l'entraînement de notre réseau de neurones est le dataset "MS_Coco" qui est trop lourd pour être uploadé sur GitHub. C'est pourquoi nous laissons ce dossier néanmoins présent pour signaler à d'autres personnes que c'est effectivement cette base de données qui a été utilisée!

Commande pour récupérer la base MS_Coco :

 wget http://images.cocodataset.org/zips/train2017.zip

Pour extraire le fichier zip dans le bon dossier une fois le téléchargement terminé:
 mkdir MS_Coco
 unzip train2017.zip -j -d /MS_Coco

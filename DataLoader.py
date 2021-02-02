import numpy as np
from PIL import Image
from glob import glob


class DataLoader():
    def __init__(self,dossier,resolution=(224,224)):
        self.dossier= dossier
        self.resolution= resolution
        self.en_entrainement= False
        self.batch_size=1
        self.facteur_reduction=2


    def entrainement(self): # méthode à invoquer pour signaler que ce sont des données d'entrainement
        self.en_entrainement=True
    
    def test(self): # méthode à invoquer pour signaler des données de test
        self.en_entrainement=False
    
    def load_data(self):
        x,y = self.resolution
        x_reduit,y_reduit= int(x/self.facteur_reduction), int(y/self.facteur_reduction)
        
        images_lowres=[]
        images_highres=[]
        
        chemins = glob("./datasets/"+self.dossier+"/*.jpg") # recup le chemin ici avec quelque chose , ptet avec glob
        
        while(len(images_lowres)<self.batch_size or len(images_highres)<self.batch_size): # On va recommencer si des images sont droppées 
            batch_chemins_images = np.random.choice(chemins,size=self.batch_size)

            for chemin in batch_chemins_images:
                
                try: # en cas de problème de chargement de l'image on va sortir de la boucle for et laisser le while recommencer
                    img = np.array(Image.open(chemin))
                except: 
                    break

                img_highres= np.array(Image.fromarray(img).resize((x,y)),dtype=float)
                img_lowres= np.array(Image.fromarray(img).resize((x_reduit,y_reduit)),dtype=float)

                #print("taille lowres: ",img_lowres.shape, "taille highres: " , img_highres.shape)

                #Data augmenting : de temps en temps on va flip horizontalement les images aléatoirement
                #resize pour redimensionner les images à la même taille
                #permet d'avoir de meilleurs résultats apparemment

                if not self.en_entrainement and np.random.random()<0.5:
                    img_highres=np.fliplr(img_highres)
                    img_lowres=np.fliplr(img_lowres)

                # Introduire une condition pour éliminer les images de mauvaise dimension
                if len(img_highres.shape)>=3:
                    images_highres.append(img_highres)

                if len(img_lowres.shape)>=3:
                    #print(img_lowres.shape)
                    images_lowres.append(img_lowres)
        
        #normalisation des données : pour avoir des valeurs de pixel entre -1 et +1
        images_highres=np.array(images_highres)/255.0*2.0-1.0
        images_lowres=np.array(images_lowres)/255.0*2.0-1.0
        return images_highres,images_lowres




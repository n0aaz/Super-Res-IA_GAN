import numpy as np
from pillow import Image
from glob import glob3


class DataLoader():
    def __init__(self,dossier,resolution=(128,128)):
        self.dossier= dossier
        self.resolution= resolution
        self.en_entrainement= False
        self.batch_size=1
        self.facteur_reduction=4

    def entrainement(self): # méthode à invoquer pour signaler que ce sont des données d'entrainement
        self.en_entrainement=True
    
    def test(self): # méthode à invoquer pour signaler des données de test
        self.en_entrainement=False
    
    def load_data(self):
        chemins = './datasets/'+self.dossier+'/' # recup le chemin ici avec quelque chose , ptet avec glob
        
        batch_chemins_images = np.random.choice(chemin,size=self.batch_size)
        
        images_lowres=[]
        images_highres=[]
        
        for chemin in batch_chemins_images:
            img = np.array(Image.open(chemin))
            x,y = self.resolution
            x_reduit,y_reduit= int(x/self.facteur_reduction), int(y/self.facteur_reduction)

            img_highres= np.array(Image.fromarray(img).resize((x,y)),dtype=float)
            img_lowres= np.array(Image.fromarray(img).resize((x_reduit,y_reduit)))

            #Data augmenting : de temps en temps on va flip horizontalement les images aléatoirement
            #permet d'avoir de meilleurs résultats apparemment

            if not self.en_entrainement and np.random.random()<0.5:
                images_highres=np.fliplr(img_highres)
                images_highres=np.fliplr(img_lowres)
            
            images_highres.append(img_highres)
            images_lowres.append(img_lowres)
        
        #normalisation des données : pour avoir des valeurs de pixel entre -1 et +1
        images_highres=np.array(images_highres)/255.0*2.0-1.0

        return images_highres,images_lowres




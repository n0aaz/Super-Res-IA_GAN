### Notre SRGAN est principalement adapté pour traiter des images de 224*224 
### Néanmoins la différence en qualité à ce niveau n'est pas absolument incroyable
### C'est pourquoi on va créer une fonction qui utilise le modèle compilé en
### découpant chaque image que l'on donne en entrée en plein de petits "blocs" de 
### 224*224 puis en réassemblant

import numpy as np
from PIL import Image
#from SRGAN import SRGAN as srgan
import matplotlib.pyplot as plt
from SRGAN import SRGAN as srgan

class ImageSuperRes():
    def __init__(self,tx=224,ty=224):
        self.tailleX = tx
        self.tailleY = ty
    def decoupe_images(self,cheminSource):
        x=self.tailleX
        y=self.tailleY
        try:
            img = np.array(Image.open(cheminSource))
        except : 
            print("Erreur: fichier non trouvé")
            return []

        nbColonnes=img.shape[0]//x
        nbLignes=img.shape[1]//y

        tableau = np.array(
            [[img[
                k*x:(k+1)*x,
                l*y:(l+1)*y,
                :] 
                for l in range(nbLignes)] 
                for k in range(nbColonnes)]
                    ,dtype=object)

        return tableau

    def reconstitue_image(self,tableau):

        tX = tableau[0,0].shape[0]
        tY = tableau[0,0].shape[1]
        print(tX,tY)
        x= len(tableau)*tX
        y= len(tableau[0])*tY
        
        image= np.zeros((x,y,3),dtype='float')
        
        for k in range(len(tableau)):
            for l in range(len(tableau[0])):
                print(tableau[k,l][:10,:10])
                image[k*tX:(k+1)*tX, l*tY:(l+1)*tY,:]= tableau[k,l]
        
        return image

    def superResolution(self, cheminSource):
        sr=srgan()
        sr.charger_modeles()
        
        miniImages= self.decoupe_images(cheminSource)
        
        miniImages = ((miniImages/255.0)-0.5)*2.0 ## ramener entre -1 et 1

        predictions = np.array([[
            255*0.5 *( sr.generateur.predict(
                np.array([miniImages[k,l]],dtype="float32")
            )[0]+1 )
            for k in range(len(miniImages)) ]
            for l in range(len(miniImages[0]))
            ],dtype=object)
        
        self.reconstitue_image(predictions)
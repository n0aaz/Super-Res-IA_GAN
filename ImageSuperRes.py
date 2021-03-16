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
import datetime

class ImageSuperRes():
    def __init__(self,tx=112,ty=112):
        self.tailleX = tx
        self.tailleY = ty
        self.dequadrillage= False
    def decoupe_images(self,cheminSource):
        print("Decoupe des images en blocs de ",self.tailleX,"x",self.tailleY," en cours...")
        x=self.tailleX
        y=self.tailleY
        try:
            img = np.array(Image.open(cheminSource))
        except : 
            print("Erreur: fichier non trouvé")
            return []

        nbColonnes=img.shape[0]//x
        nbLignes=img.shape[1]//y
        
        padded=np.zeros((y*(nbColonnes+1),x*(nbLignes+1),3))
        padded[:img.shape[0],:img.shape[1],:] = img[:,:,:3]
        

        tableau = [[padded[
                k*x:(k+1)*x,
                l*y:(l+1)*y,
                :] 
                for l in range(nbLignes+1)] 
                for k in range(nbColonnes+1)]
        
        tableau = np.array(tableau,dtype=object)
        
        print("Decoupe avec succes de l'image en ",len(tableau)*len(tableau[0])," blocs!")
        return tableau,img.shape

    def reconstitue_image(self,tableau,origShape):
        
        print("Reconstitution de ", len(tableau), "petites images en cours...")
        tX = tableau[0,0].shape[0]
        tY = tableau[0,0].shape[1]
        x= len(tableau)*tX
        y= len(tableau[0])*tY
        
        image= np.zeros((x,y,3),dtype='float')
        
        for k in range(len(tableau)):
            for l in range(len(tableau[0])):
                #print(tableau[k,l][:10,:10])
                image[k*tX:(k+1)*tX, l*tY:(l+1)*tY,:]= tableau[k,l]
        
        print("Reconstitution effectuée!, taille finale: ",origShape)
        return image[0:origShape[0],0:origShape[1],:]

    def superResolution(self, cheminSource):
        sr=srgan()
        sr.charger_generateur()
        
        debut = datetime.datetime.now()
        miniImages,shapeOriginal= self.decoupe_images(cheminSource)
        
        miniImages = ((miniImages/255.0)-0.5)*2.0 ## ramener entre -1 et 1

        predictions = np.array([
            255*0.5 *( sr.generateur.predict(
                np.array(miniImages[k],dtype="float32")
            )+1 )
            for k in range(len(miniImages)) ],dtype=object)
        
        if self.dequadrillage :
            artefact = sr.generateur.predict(-np.ones((1,self.tailleX,self.tailleY,3)))[0]
            print("Artefact: ",artefact)
            for prediction in predictions:
                prediction -= artefact
        print("temps de génération: ", datetime.datetime.now()-debut)
        return self.reconstitue_image(predictions,shapeOriginal)
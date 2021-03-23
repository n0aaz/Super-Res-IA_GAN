### Notre SRGAN est principalement adapté pour traiter des images de 224*224 
### Néanmoins la différence en qualité à ce niveau n'est pas absolument incroyable
### C'est pourquoi on va créer une fonction qui utilise le modèle compilé en
### découpant chaque image que l'on donne en entrée en plein de petits "blocs" de 
### 224*224 puis en réassemblant

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from SRGAN import SRGAN as srgan
import datetime

class ImageSuperRes():
    def __init__(self,tx=112,ty=112):
        self.tailleX = tx
        self.tailleY = ty
        self.dequadrillage= False
        self.facteur_upscaling= 4
        self.alpha_dequadrillage = 0.75
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
        return tableau,(img.shape[0],img.shape[1])

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

    def supprimer_quadrillage(self,img_lowres,img_superres):
        print("Amélioration de la qualité de l'image finale...")
        x_superres,y_superres = img_superres[:,:,0].shape
        x_lowres,y_lowres= img_lowres[:,:,0].shape
        # positions du centre de chaque image
        xMilieu_superres,yMilieu_superres = x_superres//2,y_superres//2
        xMilieu_lowres,yMilieu_lowres= x_lowres//2,y_lowres//2

        #### Calcul des transformées de fourier/fft : ####
        # fft de la forme fft_lowres[x,y,r/g/b]
        fft_lowres = np.array([np.fft.fft2(img_lowres[:,:,k]) for k in range(3)])
        # on shift chacune des fft: on décale fft(0) (la moyenne) au centre de l'image
        fftShifted_lowres = np.fft.fftshift(fft_lowres)

        fft_superres = np.array([np.fft.fft2(img_superres[:,:,k]) for k in range(3)])
        fftShifted_superres = np.fft.fftshift(fft_superres)

        #### Injection de la fft de l'image lowres dans la fft superres ####
        # xxxxxxxxxxxxxxxxxxxx
        # xxx00000000000000xxx
        # xxx00000000000000xxx      - x : fft de la superresolution
        # xxx00000000000000xxx      - 0 : fft de la basse résolution
        # xxx00000000000000xxx
        # xxx00000000000000xxx
        # xxxxxxxxxxxxxxxxxxxx

        # position des quatres coins du carré d'insertion
        positions = np.array([
                        [xMilieu_superres-xMilieu_lowres , xMilieu_superres-xMilieu_lowres+x_lowres],
                        [yMilieu_superres-yMilieu_lowres, yMilieu_superres-yMilieu_lowres+y_lowres]
                    ])
        # explication du calcul : notre image de la superresolution contient 16* plus de pixels soit 4*4
        # l'énergie de sa fft est donc 16x supérieure à celle de la fft de basse résolution
        # c'est pourquoi pour équilibrer on multiplie la fft de la basse résolution par la même quantité
        # pour conserver une uniformité des amplitudes lorsqu'on insère l'un dans l'autre !
        # ensuite on fait une moyenne pondérée entre les deux fft sur la zone intérieure:
        # centre(nouvelle fft) = alpha * fft_lowres + (1-alpha) * centre(fft_superres)
        # de cette manière on peut contrôler la quantité de détails que l'on conserve!
        fftShifted_superres[:,positions[0,0]:positions[0,1],positions[1,0]:positions[1,1]]= (self.facteur_upscaling**2)*self.alpha_dequadrillage* fftShifted_lowres + \
fftShifted_superres[:,positions[0,0]:positions[0,1],positions[1,0]:positions[1,1]]*(1-self.alpha_dequadrillage)

        # contient des zéros pour l'instant mais on va le remplir
        img_corrected= np.zeros(img_superres.shape)

        #### Reconstitution à partir de la fft modifiée ####
        # 1- reshifter la fft au bon endroit (centre à x=0,y=0)
        superres_reconstruction = np.fft.ifftshift(fftShifted_superres)
        # 2- fft inverse
        superres_reconstruction = np.fft.ifft2(superres_reconstruction)
        # 3- module : la ffti renvoie des complexes
        superres_reconstruction = np.abs(superres_reconstruction)
        # 4- mise en forme et renormalisation de chaque couleur reconstruite
        max_reconstruction = np.max(superres_reconstruction)
        img_corrected[:,:,0]= (superres_reconstruction[0])/max_reconstruction
        img_corrected[:,:,1]=(superres_reconstruction[1])/max_reconstruction
        img_corrected[:,:,2]=(superres_reconstruction[2])/max_reconstruction

        print("Effectué !")
        return img_corrected

    def superResolution(self, cheminSource):
        sr=srgan()
        sr.charger_generateur()
        
        debut = datetime.datetime.now()
        miniImages,shapeOriginal= self.decoupe_images(cheminSource)
        image_lowres = np.asarray(Image.open(cheminSource))
        shapeUpscaled= shapeOriginal[0]*self.facteur_upscaling, shapeOriginal[1]*self.facteur_upscaling
        
        miniImages = ((miniImages/255.0)-0.5)*2.0 ## ramener entre -1 et 1

        predictions = np.array([
            255*0.5 *( sr.generateur.predict(
                np.array(miniImages[k],dtype="float32")
            )+1 )
            for k in range(len(miniImages)) ],dtype=object)
        
        if self.dequadrillage : # déquadrillage en 3 étapes
            # 1 - générer le profil type de l'impact de la superresolution
            # sur une image vide
            artefact = sr.generateur.predict(-np.ones((1,self.tailleX,self.tailleY,3)))[0]
            #print("Artefact: ",artefact)

            # 2 - soustraire ce pattern à chaque bout d'image 
            #for prediction in predictions:
            #    prediction -= artefact
            
            # 3 - terminer la suppression du quadrillage grace a la methode
            # des fft ! 
            print("temps de génération: ", datetime.datetime.now()-debut)
            return self.supprimer_quadrillage(
                image_lowres,
                self.reconstitue_image(predictions,shapeUpscaled)
                )
        else : 
            print("temps de génération: ", datetime.datetime.now()-debut)
            return self.reconstitue_image(predictions,shapeUpscaled)
# Liste des imports dont on pourra éventuellement avoir besoin

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import datetime
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np
import os

from DataLoader import DataLoader
from Generator import Generator
from Discriminator import Discriminator
from Vgg import Vgg

class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        
        self.facteur_reduction=4

        self.hr_height = 448 #self.lr_height*4   # High resolution height
        self.hr_width = 448 #self.lr_width*4     # High resolution width
        self.lr_height = self.hr_height//self.facteur_reduction #64                 # Low resolution height
        self.lr_width = self.hr_width//self.facteur_reduction #64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        self.adam_beta1= 0.9
        self.adam_beta2=0.999
        self.adam_epsilon=1e-08
        self.adam_learning_rate= 1e-04
        
        self.vgg_loss_factor= 1#/12.75 0.006 , mais mauvais résultats
        
        # Introduction d'une fonction d'autosauvegarde qui va 
        # sauvegarder le modèle à intervalles réguliers de génération
        # ainsi qu'un "aperçu" de la sortie du générateur!
        self.auto_sauvegarde = False
        self.intervalle_sauvegarde= 50
        
        print("lr_shape,hrshape=",self.lr_shape, self.hr_shape)

        self.train_sample_interval=50
        self.train_batch_size=16
        

        #self.optimizer=Adam(0.0002, 0.5)
        self.optimizer= Adam(learning_rate=self.adam_learning_rate,beta_1=self.adam_beta1,beta_2=self.adam_beta2,epsilon=self.adam_epsilon)
        
        self.vgg = Vgg(self.hr_shape)
        self.vgg = self.vgg.build()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])        

        # Configure data loader
        self.dataset_name = 'MS_Coco'
        self.data_loader = DataLoader(dossier=self.dataset_name,
                                      resolution=(self.hr_height, self.hr_width))
        self.data_loader.batch_size=self.train_batch_size
        self.data_loader.facteur_reduction=self.facteur_reduction
        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)    

        self.discriminateur= Discriminator(self.hr_shape).build()
        self.discriminateur.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])

        self.generateur= Generator(self.lr_shape).build()

        # Images (entrées) haute résolution & basse résolution
        img_highres = Input(shape=self.hr_shape)
        img_lowres = Input(shape=self.lr_shape)

        # Haute résolution générée par le générateur:
        gen_highres = self.generateur(img_lowres)

        # Features de l'image du générateur
        gen_features= self.vgg(gen_highres)

        
        self.discriminateur.trainable =False

        validite = self.discriminateur(gen_highres)
        self.combinaison = Model([img_lowres,img_highres],[validite,gen_features])
        self.combinaison.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1*self.vgg_loss_factor],
                              optimizer=self.optimizer)


    def train_discriminator(self):
        self.data_loader.batch_size=self.train_batch_size
        #self.data_loader.facteur_reduction=self.facteur_reduction
        
        highres,lowres = self.data_loader.load_data()
        
        print("batch size= ",self.train_batch_size,self.data_loader.batch_size)
        print(len(lowres[0]),len(highres[0]))
        
        # image haute résolution générée 
        
        print("train_generator line 90 , shape of lowres :",highres.shape)

        highres_genere= self.generateur.predict(lowres)

        # association des sorties à ce qu'on veut : 
        # une image originale -> le discriminateur retourne 1
        # une image générée -> le discriminateur retourne 0

        is_original= np.ones((self.train_batch_size,)+self.disc_patch)
        is_generated= np.zeros((self.train_batch_size,)+self.disc_patch)

        #entrainement du discriminateur sur les images originales puis générées
        # et calcul du "loss" c'est à dire la fonction de perte 
        discriminateur_loss_original = self.discriminateur.train_on_batch(highres,is_original)
        discriminateur_loss_genere = self.discriminateur.train_on_batch(highres_genere,is_generated)
        # la perte totale est la moyenne des deux pertes
        discriminateur_loss_total = 0.5*np.add(discriminateur_loss_original,discriminateur_loss_genere)
        return discriminateur_loss_total
    
    def train_generator(self):
        self.data_loader.batch_size=self.train_batch_size
        self.data_loader.facteur_reduction=self.facteur_reduction
        highres,lowres = self.data_loader.load_data()

        # le but du générateur est de tendre vers le modèle où le discriminateur
        # ne renvoie que des 1 (tout est original)
        is_original=np.ones((self.train_batch_size,)+self.disc_patch)

        # vgg va caracteriser les images, ces caractéristiques sont un critere d'entrainement
        caract_images_reelles = self.vgg.predict(highres)

        generateur_loss = self.combinaison.train_on_batch([lowres,highres],[is_original,caract_images_reelles])
        # entrainement de la combinaison des deux pour avoir la perte du générateur
        # lowres (entree) ->generateur-> fausse image -> discriminateur -> validite qu'on compare a la validité idéale 
        # lowres (entree) ->generateur -> fausse image -> VGG19 -> caracteristique de la fausse image qu'on compare 
        # aux caracteristiques de la vraie image highres

        return generateur_loss

    def train(self,generations):
        
        #debut chrono pour mesurer le temps d'entrainement
        debut=datetime.datetime.now()

        for k in range(generations):

            debutgeneration=datetime.datetime.now()
            disc_loss = self.train_discriminator()
            gen_loss = self.train_generator()

            temps = datetime.datetime.now()-debut
            tempsgeneration = datetime.datetime.now() -debutgeneration
            
            # pour qu'on voie sur le terminal l'avancée
            print("Generation n°:"+str(k)+"\nDuree: "+str(tempsgeneration)+"\nDuree Totale:"+str(temps)+"\n")
            print("Perte (loss) du générateur: ",gen_loss,"\nPerte du discriminateur: ",disc_loss)
            
            if((self.auto_sauvegarde) and ((k+1)%self.intervalle_sauvegarde==0)):
                print("--- Intervalle de sauvegarde atteint ---\n")
                print("Nous en sommes à l'échantillon n°",k)
                self.echantillon_images(k)
                self.sauvegarde_modeles()
                print("--- Sauvegarde terminée ----------------\n")
                

    # Pour calculer le PSNR entre deux images type sortie de réseau de neurones
    def calculate_psnr(self,image1,image2):
        # Attention: les valeurs de image1 et image2 doivent être entre -1 et +1
        from math import log10,sqrt
        image1_Fullscale = (0.5 * image1 + 0.5)*255.0 
        image2_Fullscale = (0.5 * image2 + 0.5)*255.0 
        mse= np.mean((image1_Fullscale-image2_Fullscale)**2)
        return 20*log10(255.0/sqrt(mse))

        
    def echantillon_images (self, generation) :#generation=epoch
    
        os.makedirs('images/%s/training' % self.dataset_name, exist_ok = True)
        #bibliothèque os (interaction systeme d'exploitation) : création répertoire
    
        #Récupération des images
        self.data_loader.entrainement=True
        self.data_loader.batch_size=1
        highres, lowres = self.data_loader.load_data()
        highres_genere = self.generateur.predict(lowres)
    
        # Redimensionnement
        lowres = 0.5 * lowres + 0.5
        highres_genere = 0.5 * highres_genere + 0.5
        highres = 0.5 * highres + 0.5

        # Calcul du PSNR:
        psnr=self.calculate_psnr(highres,highres_genere)
        psnr=round(psnr,2)

        #Sauvergarde des images HR générées et des images HR
        ligne, colonne = 1,3
        titres = ['Basse Résolution',f"Haute résolution générée\nPSNR={psnr} dB",'Haute résolution originale']
        fig, axs = plt.subplots(ligne,colonne)
        fig.set_size_inches(12,7)
        compteur = 0
        for i in range(ligne) :
            for colonne, image in enumerate ([lowres,highres_genere,highres]):
                axs[colonne].imshow(image[i])
                xImage,yImage,zImage = image[i].shape
                xOrigine,yOrigine=0.3*xImage,0.3*yImage
                largeurX,largeurY=xImage*0.15,yImage*0.15
                axins = zoomed_inset_axes(axs[colonne], 3, loc=1) # zoom = 4
                axins.imshow(image[i], interpolation="nearest",origin="lower")
                axins.set_xlim(xOrigine,xOrigine+largeurX)
                axins.set_ylim(yOrigine+largeurY,yOrigine)
                axins.axis('off')
                mark_inset(axs[colonne], axins, loc1=2, loc2=4, fc="none", ec="r")
                axs[colonne].set_title(titres[colonne])
                #axs[colonne].axis('off')
            compteur += 1
        fig.savefig('images/%s/training/%d.png' % (self.dataset_name, generation))
        plt.close()
            
    def sauvegarde_modeles(self,dossier="Model/"):
        # Fonction pour pouvoir sauvegarder le modèle une fois enregistré, pour ne pas avoir à recalculer
        # instructions ici : https://www.tensorflow.org/guide/keras/save_and_serialize
        
        # -- Sauvegarde générateur --
        self.generateur.save(dossier+"Generateur")
        # -- Sauvegarde discriminateur --
        self.discriminateur.save(dossier+"Discriminateur")
        # -- Sauvegarde modèle combiné --
        self.combinaison.save(dossier+"modeleCombine")
        
        print("Modèles sauvegardés avec succès!")
        
    def charger_modeles(self,dossier="Model/"):
        # Fonction pour pouvoir charger le modèle une fois enregistré, pour ne pas avoir à recalculer
        # instructions ici : https://www.tensorflow.org/guide/keras/save_and_serialize
        
        print("Chargement du modèle depuis ",dossier," en cours ...")
        # -- Charger générateur --
        self.generateur= tf.keras.models.load_model(dossier+"Generateur")
        # -- Charger discriminateur --
        self.discriminateur= tf.keras.models.load_model(dossier+"Discriminateur")
        # -- Charger modèle combiné --
        self.combinaison= tf.keras.models.load_model(dossier+"modeleCombine")

        # Les modèles que l'on avait sauvegardé précédemment n'étaient pas compilés
        # On doit les compiler avant de pouvoir les utiliser.
        print("Compilation du discriminateur...")
        self.discriminateur.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        print("Compilation du modèle combiné...")
        self.combinaison.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1*self.vgg_loss_factor],
                              optimizer=self.optimizer)
        print("Modèle chargé !\n")
        
    def charger_generateur(self,dossier="Model/"):
        ## Pareil que précédemment mais on charge que le générateur pour se faciliter la vie vu qu'on veut juste 
        ## générer une image à l'aide du générateur
        
        print("Chargement du générateur depuis ",dossier," en cours ...")
        # -- Charger générateur --
        self.generateur= tf.keras.models.load_model(dossier+"Generateur")
        
        print("Générateur chargé!\n")
        

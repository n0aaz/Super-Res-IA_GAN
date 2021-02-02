# Liste des imports dont on pourra éventuellement avoir besoin
import scipy
import tensorflow as tf
from tensorflow.keras.datasets import mnist
#from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from DataLoader import DataLoader
import numpy as np
import os

from Generator import Generator
from Discriminator import Discriminator
from Vgg import Vgg

import tensorflow.keras.backend as K

class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 224//4 #64                 # Low resolution height
        self.lr_width = 224//4 #64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = 224 #self.lr_height*4   # High resolution height
        self.hr_width = 224 #self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        print("lr_shape,hrshape=",self.lr_shape, self.hr_shape)

        self.train_sample_interval=50
        self.train_batch_size=16
        

        self.optimizer=Adam(0.0002, 0.5)

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
                              loss_weights=[1e-3, 1],
                              optimizer=self.optimizer)


    def train_discriminator(self):
        self.data_loader.batch_size=self.train_batch_size
        highres,lowres = self.data_loader.load_data()
        
        print("batch size= ",self.train_batch_size,self.data_loader.batch_size)
        print(len(lowres),len(highres))
        
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
            print("Generation n°:"+str(k)+"\nDuree: "+str(tempsgeneration)+"\nDuree Totale:"+str(temps))
        
    def echantillon_images (self, generation) :#generation=epoch
    
        os.makedirs('images/%s/training' % self.dataset_name, exist_ok = True)
        #bibliothèque os (interaction systeme d'exploitation) : création répertoire
    
        #Récupération des images
        self.data_loader.entrainement=True
        self.data_loader.batch_size=2
        highres, lowres = self.data_loader.load_data()
        highres_genere = self.generateur.predict(lowres)
    
        # Redimensionnement
        lowres = 0.5 * lowres + 0.5
        highres_genere = 0.5 * highres_genere + 0.5
        highres = 0.5 * highres + 0.5
    
        #Sauvergarde des images HR générées et des images HR
        ligne, colonne = 2,2
        titres = ['Haute résolution générée','Haute résolution originale']
        fig, axs = plt.subplots(ligne,colonne)
        fig.set_size_inches(12,8)
        compteur = 0
        for i in range(ligne) :
            for colonne, image in enumerate ([highres_genere,highres]):
                axs[i, colonne].imshow(image[i])
                axs[i, colonne].set_title(titres[colonne])
                axs[i, colonne].axis('off')
            compteur += 1
        fig.savefig('images/%s/training/%d.png' % (self.dataset_name, generation))
        plt.close()

        #Sauvegarde des images BR pour la comparaison
        for j in range (ligne) :
            fig = plt.figure()
            plt.imshow(lowres[i])
            
            fig.savefig('images/%s/training/%d_lowres%d.png' % (self.dataset_name, generation, i))
            
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
                              loss_weights=[1e-3, 1],
                              optimizer=self.optimizer)
        print("Modèle chargé !\n")
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

        self.train_sample_interval=50
        self.train_batch_size=1

        optimizer=Adam(0.0002, 0.5)

        self.vgg = Vgg(self.hr_shape).build()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])        

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.data_loader = DataLoader(dossier=self.dataset_name,
                                      resolution=(self.hr_height, self.hr_width))
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)    

        self.discriminateur= Discriminator(self.hr_shape).build()
        self.discriminateur.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generateur= Generator(self.lr_shape).build()

        # Images (entrées) haute résolution & basse résolution
        img_highres = Input(shape=self.hr_shape)
        img_lowres = Input(shape=self.lr_shape)

        # Haute résolution générée par le générateur:
        gen_highres = self.generator(img_lowres)

        # Features de l'image du générateur
        gen_features= self.vgg(gen_highres)

        
        self.discriminateur.trainable =False

        validite = self.discriminateur(gen_highres)
        self.combinaison = Model([img_lowres,img_highres],[validite,gen_features])
        self.combinaison.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)


    def train_discriminator(self):
        highres,lowres = self.data_loader.load_data(self.train_batch_size)

        # image haute résolution générée 
        highres_genere= self.generator.predict(lowres)

        # association des sorties à ce qu'on veut : 
        # une image originale -> le discriminateur retourne 1
        # une image générée -> le discriminateur retourne 0

        is_original= np.ones((self.train_batch_size,)+ self.disc_patch)
        is_generated= np.zeros((self.train_batch_size)+self.disc_patch)

        #entrainement du discriminateur sur les images originales puis générées
        # et calcul du "loss" c'est à dire la fonction de perte 
        discriminateur_loss_original = self.discriminateur.train_on_batch(highres,is_original)
        discriminateur_loss_genere = self.discriminateur.train_on_batch(highres_genere,is_generated)
        # la perte totale est la moyenne des deux pertes
        discriminateur_loss_total = 0.5*np.add(discriminateur_loss_original,discriminateur_loss_genere)
        return discriminateur_loss_total
    
    def train_generator(self):
        highres,lowres = self.data_loader.load_data(self.train_batch_size)

        # le but du générateur est de tendre vers le modèle où le discriminateur
        # ne renvoie que des 1 (tout est original)
        is_original=np.ones((batch_size,)+self.disc_patch)

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
            disc_loss = train_discriminator()
            gen_loss = train_generator()

            temps = datetime.datetime.now()-debut
            tempsgeneration = datetime.datetime.now() -debutgeneration
            
            # pour qu'on voie sur le terminal l'avancée
            print("Generation n°:"+str(k)+"\nDuree: "+str(tempsgeneration)+"\nDuree Totale:"+str(temps))
        
    def echantillon_images (self, generation) :#generation=epoch
    
        os.mkdir('images/%s/training' % self.dataset_name, exist_ok = True)
        #bibliothèque os (interaction systeme d'exploitation) : création répertoire
    
        #Récupération des images
        self.data_loader.entrainement=True
        self.data_loader.batch_size=2
        highres, lowres = self.data_loader.load_data(batch_size=2, is_testing=True)
        highres_genere = self.generateur.predict(lowres)
    
        # Redimensionnement
        lowres = 0.5 * lowres + 0.5
        highres_genere = 0.5 * highres + 0.5
        highres = 0.5 * highres + 0.5
    
        #Sauvergarde des images HR générées et des images HR
        ligne, colonne = 2,2
        titres = ['Haute résolution générée','Haute résolution originale']
        figure, axes = plt.subplots(ligne,colonne)
        compteur = 0
        for i in range(ligne) :
            for colonne, image in enumerate ([highres_genere,highres]):
                axs[ligne, colonne].imshow(image[ligne])
                axs[ligne, colonne].set_title(titles[colonne])
                axs[ligne, colonne].axis('off')
            compteur += 1
        fig.savefig('images/%s/training/%d.png' % (self.dataset_name, generation))
        plt.close()

        #Sauvegarde des images BR pour la comparaison
        for j in range (ligne) :
            figure = plt.figure()
            plt.imshow(lowres[i])
            
            fig.savefig('images/%s/training/%d_lowres%d.png' % (self.dataset_name, generation, i))
            plt.close()


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
from data_loader1 import DataLoader
import numpy as np
import os

import Generator,Discriminator,Vgg

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

        optimizer=Adam(0.0002, 0.5)

        self.vgg = Vgg(hr_shape).build()
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

        self.discriminateur= Discriminator(hr_shape).build()
        self.discriminateur.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generateur= Generator(lr_shape).build()

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



    def build(self):
        pass
        
# Liste des imports
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

import tensorflow.keras.backend as K

class Discriminator():
    
    def __init__(self, lr_shape,hr_shape) :
        
        #Construction et Compilation du discriminateur
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',optimizer=optimizer, metrics=['accuracy'])
    
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)
        
    #Discriminateur détermine validité de la génération d'image HR
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1],optimizer=optimizer)
    
    def dicriminateur_bloc(layer_entree, filtres,bn=True) :
        #Ecriture du bloc réseau décrit dans la doc
        d = Conv2D(filtres, kernel=3,strides=1,padding='same')(layer_entree)
        d = LeakyReLU(alpha = 0.2)(d)
        #Utilisation de la fonction d'activation LeakyReLU + efficace que ReLU "
        
        if bn :
            d = BatchNormalization(momentum=0.8)(d)
        return d
        
       
    def discriminateur_réseau(self) :
       
        d0 = Input(shape=self.hr_shape)

        d1 = discriminateur_bloc(d0, self.df, bn=False)
        d2 = discriminateur_bloc(d1, self.df, strides=2)
        d3 = discriminateur_bloc(d2, self.df*2)
        d4 = discriminateur_bloc(d3, self.df*2, strides=2)
        d5 = discriminateur_bloc(d4, self.df*4)
        d6 = discriminateur_bloc(d5, self.df*4, strides=2)
        d7 = discriminateur_bloc(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)
       
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        
        return d10
    
    def Build_Discriminateur() :
        
        validity = Dense(1, activation='sigmoid')(d10)
    
        return Model(d0, validity)

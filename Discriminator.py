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

    def Dicriminateur_bloc(layer_entree, filtres,bn=True) :
        "Ecriture du bloc réseau décrit dans la doc"
        d = Conv2D(filtres, kernel=3,strides=1,padding='same')(layer_entree)
        "Voir paramètres de Conv2D"
        d = LeakyReLU(alpha = 0.2)(d)
        "Utilisation de la fonction d'activation LeakyReLU + efficace que ReLU "
        
        if bn :
            d = BatchNormalization(momentum=0.8)(d)
        return d
        
       
    def Discriminateur_réseau(self) :
       
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)
       
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        
        return d10
    
    def Build_Discriminateur() :
        
        validity = Dense(1, activation='sigmoid')(d10)
    
            return Model(d0, validity)

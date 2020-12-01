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

import tensorflow.keras.backend as K

class Generator():
    def __init__(self,lrshape):
        self.nbfiltres=64
        self.lr_shape=lrshape
        #nombre de blocs résiduels
        self.nb_residual = 16
        # normalement 3 canaux : R,G,B (image), contenu à la fin de 
        # notre shape
        _,_,self.channels = self.lr_shape
    
    def residual_block(self,layer_input,filtres):
        d = Conv2D(filtres, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filtres, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d
    
    def preresidual_block(self,layer_input):
        preresidual = Conv2D(64, kernel_size=9, strides=1, padding='same')(image_lowres)
        preresidual = Activation('relu')(preresidual)
        return preresidual
        
    def deconv2d(self,layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    def build(self):
        image_lowres= Input(shape=self.lr_shape)
        # Bloc préresiduel
        preresidual = preresidual_block(image_lowres)

        # Blocs résiduels
        residual= residual_block(preresidual,self.nbfiltres)
        for k in range(self.nb_residual-1):
            residual=residual_block(residual,self.nbfiltres)
        
        # Bloc postrésiduel
        postresidual


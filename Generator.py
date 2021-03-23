# Liste des imports dont on pourra éventuellement avoir besoin
#import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model

import numpy as np

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
        d0 = Conv2D(filtres, kernel_size=3, strides=1, padding='same')(layer_input)
        d1 = Activation('relu')(d0)
        d2 = BatchNormalization(momentum=0.8)(d1)
        d3 = Conv2D(filtres, kernel_size=3, strides=1, padding='same')(d2)
        d4 = BatchNormalization(momentum=0.8)(d3)
        d5 = Add()([d4, layer_input])
        return d5
    
    def preresidual_block(self,layer_input):
        preresidual0 = Conv2D(64, kernel_size=9, strides=1, padding='same')(layer_input)
        preresidual1 = Activation('relu')(preresidual0)
        return preresidual1

    def postresidual_block(self,layer_input,preresidual_input):
        postresidual0 = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
        postresidual1 = BatchNormalization(momentum=0.8)(postresidual0)
        postresidual2 = Add()([postresidual1, preresidual_input])
        return postresidual2

        
    def deconv2d(self,layer_input):
        """Layers used during upsampling"""
        u0 = UpSampling2D(size=2)(layer_input)
        u1 = Conv2D(256, kernel_size=3, strides=1, padding='same')(u0)
        u2 = Activation('relu')(u1)
        return u2

    def build(self):
        image_lowres= Input(shape=self.lr_shape)
        # Bloc préresiduel
        preresidual = self.preresidual_block(image_lowres)

        # Blocs résiduels
        residual= self.residual_block(preresidual,self.nbfiltres)
        for k in range(self.nb_residual-1):
            residual=self.residual_block(residual,self.nbfiltres)
        
        # Bloc postrésiduel
        postresidual=self.postresidual_block(residual,preresidual)

        # Upsampling
        upsample0 = self.deconv2d(postresidual)
        upsample1 = self.deconv2d(upsample0) # on augmente encore un coup la résolution

        # Sortie finale
        sortie_highres=Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(upsample1)

        # On doit retourner un modèle de "bloc" tensorflow représenté comme une boite noire
        # avec une entrée et une sortie.
        return Model(image_lowres,sortie_highres) 

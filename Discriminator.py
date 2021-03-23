# Liste des imports
#import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np


import tensorflow.keras.backend as K

class Discriminator():
    
    def __init__(self,hrshape) :
        self.df = 64 #nombre de filtres
        self.hr_shape = hrshape
    
    def discriminateur_bloc(self, layer_entree, filtres,bn=True,strides=1) :
        #Ecriture du bloc réseau décrit dans la doc
        d = Conv2D(filtres, kernel_size=3,strides=strides,padding='same')(layer_entree)
        d = LeakyReLU(alpha = 0.2)(d)
        
        if bn :
            d = BatchNormalization(momentum=0.8)(d)
        return d
        
       
    def build(self) :
       
        d0 = Input(shape=self.hr_shape)
        
        d1 = self.discriminateur_bloc(d0, self.df, bn=False)
        d2 = self.discriminateur_bloc(d1, self.df, strides=2)
        d3 = self.discriminateur_bloc(d2, self.df*2)
        d4 = self.discriminateur_bloc(d3, self.df*2, strides=2)
        d5 = self.discriminateur_bloc(d4, self.df*4)
        d6 = self.discriminateur_bloc(d5, self.df*4, strides=2)
        d7 = self.discriminateur_bloc(d6, self.df*8)
        d8 = self.discriminateur_bloc(d7, self.df*8, strides=2)
       
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)

        validity = Dense(1, activation='sigmoid')(d10)
    
        return Model(d0, validity)

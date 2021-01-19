import tensorflow as tf
import numpy as np
from keras.applications import VGG19
from keras.layers import Input
from keras.models import Model

#from tensorflow.keras.applications import VGG19
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

class Vgg():
    def __init__(self, hrshape):
        self.hr_shape = hrshape

    def build(self):
        entree = Input(shape = self.hr_shape)
        print("REGARDE ICI : ",entree.shape,self.hr_shape)
        
        vgg=VGG19(input_shape=self.hr_shape,weights="imagenet")
        
        #corrige le problème de pop from empty list mais pas sur pourquoi
        #vgg.outputs = [vgg.layers[9].output]
        
        # Récupération des "features" de l'image: des nombres qui caractérisent une image
        #print(entree)
        img_features = vgg(entree)

        return Model(entree, img_features)        
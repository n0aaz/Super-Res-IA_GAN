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
        #print("REGARDE ICI : ",entree.shape,self.hr_shape)
        
        vgg=VGG19(include_top= False, input_shape=self.hr_shape,weights="imagenet")
        for l in vgg.layers:
            l.trainable=False
        
        #corrige le problème de pop from empty list mais pas sur pourquoi
        #vgg.outputs = [vgg.layers[9].output]
        
        # Récupération des "features" de l'image: des nombres qui caractérisent une image
        #print(entree)
        img_features = vgg.get_layer('block2_conv2').output

        return Model(inputs=vgg.input, outputs=img_features)        

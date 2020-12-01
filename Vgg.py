import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

class Vgg():
    def __init__(self,hrshape):
        hr_shape = hrshape

    def build(self):
        vgg=VGG19(weights="imagenet")

        vgg.outputs = [vgg.layers[9].output]

        entree = Input(shape=self.hr_shape)
        #print("REGARDE ICI : ",self.hr_shape)

        # Récupération des "features" de l'image: des nombres qui caractérisent une image
        img_features = vgg(entree)

        return Model(entree, img_features)        
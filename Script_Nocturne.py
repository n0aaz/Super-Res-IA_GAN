### ATTENDRE QUELQUES HEURES QUE LE SERVEUR AIT FINI DE BOUGER LES IMAGES
from SRGAN import SRGAN as srgan
from tensorflow.keras.optimizers import Adam
#from time import sleep 
#sleep(3600*3)

SR=srgan()
SR.train_batch_size = 6

temps_generation = 1.9 # en secondes
temps_sauvegarde = 300 # 5min
temps_entrainement = 48*3600 # 10h

SR.auto_sauvegarde= True
SR.intervalle_sauvegarde= int(temps_sauvegarde//temps_generation) # temps par sauvegarde : environ 7s, une sauvegarde toutes les 4min

# Changement de learning rate: 

SR.adam_learning_rate/= 100
SR.optimizer= Adam(learning_rate=SR.adam_learning_rate,beta_1=SR.adam_beta1,beta_2=SR.adam_beta2,epsilon=SR.adam_epsilon)


SR.charger_modeles()
SR.train(int(temps_entrainement//temps_generation)) # 24h d'entrainement, autosauvegarde en +


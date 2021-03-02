### ATTENDRE QUELQUES HEURES QUE LE SERVEUR AIT FINI DE BOUGER LES IMAGES
from SRGAN import SRGAN as srgan
#from time import sleep 
#sleep(3600*3)

SR=srgan()
SR.train_batch_size = 5 # temps par génération : environ 3s

temps_generation = 3 # en secondes
temps_sauvegarde = 900 # 4min
temps_entrainement = 10*3600 # 10h

SR.auto_sauvegarde= True
SR.intervalle_sauvegarde= temps_sauvegarde//temps_generation # temps par sauvegarde : environ 7s, une sauvegarde toutes les 4min
SR.charger_modeles()
SR.train(temps_entrainement//temps_generation) # 24h d'entrainement, autosauvegarde en +


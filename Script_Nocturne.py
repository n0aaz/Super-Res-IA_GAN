### ATTENDRE QUELQUES HEURES QUE LE SERVEUR AIT FINI DE BOUGER LES IMAGES
from SRGAN import SRGAN as srgan
#from time import sleep 
#sleep(3600*3)

SR=srgan()
SR.train_batch_size = 16 # temps par génération : environ 2s 
SR.auto_sauvegarde= True
SR.intervalle_sauvegarde= 100 # temps par sauvegarde : environ 7s
SR.train(2400) # temps d'entrainement estimé : 4800s + 7*12s = 1h21


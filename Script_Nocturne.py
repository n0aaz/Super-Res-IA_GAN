### ATTENDRE QUELQUES HEURES QUE LE SERVEUR AIT FINI DE BOUGER LES IMAGES
from SRGAN import SRGAN as srgan
#from time import sleep 
#sleep(3600*3)

SR=srgan()
SR.train_batch_size = 32 # temps par génération : environ 3s 
SR.auto_sauvegarde= True
SR.intervalle_sauvegarde= 100 # temps par sauvegarde : environ 7s
SR.train(1200) # temps d'entrainement estimé : 3600s + 7*12s = 1h01


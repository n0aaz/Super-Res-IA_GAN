### ATTENDRE QUELQUES HEURES QUE LE SERVEUR AIT FINI DE BOUGER LES IMAGES
from SRGAN import SRGAN as srgan
#from time import sleep 
#sleep(3600*3)

SR=srgan()

SR.train(10000)
SR.sauvegarde_modeles()
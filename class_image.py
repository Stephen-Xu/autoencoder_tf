import tensorflow as tf
from classifier import classifier
import numpy as np
import scipy.io as sio
import sys


units = [24,24,960,96]   #################
act = ['leaky_relu6','tanh','linear']

cl = classifier(units,act)

#generate_classifier(std_w=3.0,euris=True,dropout=False)
#session = cl.init_network()

cl.generate_classifier() 
session = cl.init_network()
     
cl.stop_dropout()
    
cl.load_model('./converted.mdl',session=session)

mat = sio.loadmat("./single_ex.mat")
data = mat['a']

data_ = np.expand_dims(data,0)


o,c = cl.get_convolution(data)


print o
print c


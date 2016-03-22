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


o,r = get_conv(data_)




l_o = np.squeeze(o)

l_o = np.reshape(l_o,[109*109,24])

c = cl.session.run(cl.output(l_o))

c_o = np.reshape(c,[109,109,96])

c_o = np.expand_dims(c_o,0)

print np.mean(pow(c_o-r,2))


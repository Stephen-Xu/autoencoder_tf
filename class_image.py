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
    
cl.load_model('./converted_normed.mdl',session=session)

mat = sio.loadmat("./subset.mat")
data = mat['subset'].astype("float32")

mat2 = sio.loadmat("./bias")
bias = mat2['bias']

bias_m = np.ones([109,109,96])

for i in range(96):
	bias_m[:,:,i] = np.tile(bias[i],[109,109])


res_reduced = np.zeros([50,109,109,96])
res_original = np.zeros([50,109,109,96])
#data_ = np.expand_dims(data,0).astype("float32")

#data_ = data_[:,:7,:7,:3]

#print data_

for i in range(50):

	x = np.expand_dims(data[i,:,:,:],0)
	c,o,_,_ = cl.get_convolution(x,padding=[1,2,2,1])

	ori = cl.session.run(o)

	out = cl.session.run(c)

#print out.shape
#print ori.shape

	out2 = np.reshape(out,[1*109*109,24]).astype("float32")


	fin_out = cl.session.run(cl.output(out2))

	#ori2 = np.reshape(ori,[1*218*218,96])

#print fin_out.shape, fin_out

	#print np.mean((pow(fin_out-ori2,2)**0.5))

	recon = np.reshape(fin_out,[1,109,109,96])

	res_reduced[i,:,:,:] = np.add(recon,bias_m)
	res_original[i,:,:,:] = np.add(ori,bias_m)


dic = {'original':res_original,'reduced':res_reduced}

sio.savemat("first_test.mat",dic,do_compression=True)

#print fin_out
#print ori2

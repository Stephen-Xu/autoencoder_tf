import autoencoder
import math
import numpy as np
from tools.data_manipulation import batch

data = np.loadtxt("../datasets/multipie_rand_sel_space.dat")

data = data+abs(np.min(data))
data = data/np.max(data)

int_dim = 100

bat = batch.seq_batch(data,2500)

#units = [data.shape[1],int(math.ceil(data.shape[1]*1.2))+5,int(max(math.ceil(data.shape[1]/4),int_dim+2)+3),
#         int(max(math.ceil(data.shape[1]/10),int_dim+1)),int_dim]

units = [5600,100]

act = ['sigmoid']
#act = ['relu','relu','relu','relu']
auto = autoencoder.autoencoder(units,act)

auto.generate_encoder()
auto.generate_decoder()

session = auto.init_network()

auto.train(data,batch=bat,le=False,tau=1.0,session=session,n_iters=2000,display=False,noise=True,noise_level=0.015,noise_iter=20)


mid = auto.get_hidden(data,session=session)

out = auto.get_output(data,session=session)

np.save('middle.dat',mid)
np.save('output.dat',out)


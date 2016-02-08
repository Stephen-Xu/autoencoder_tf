import autoencoder
import math
import numpy as np
import sys
from tools.data_manipulation import batch

data = np.loadtxt("../datasets/multi_pie.dat")

data = data+abs(np.min(data))
data = data/np.max(data)
data = data.astype("float32")


bat = batch.rand_batch(data,1000)

#units = [data.shape[1],int(math.ceil(data.shape[1]*1.2))+5,int(max(math.ceil(data.shape[1]/4),int_dim+2)+3),
#         int(max(math.ceil(data.shape[1]/10),int_dim+1)),int_dim]

units = [5600,1100,200,int(sys.argv[1])]

act = ['sigmoid','sigmoid','sigmoid']
#act = ['relu','relu','relu','relu']
auto = autoencoder.autoencoder(units,act)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=True)
#auto.pre_train(data,n_iters=5000)

session = auto.init_network()

ic,bc = auto.train(data,n_iters=5000,record_weight=False,w_file='./pie_weights_20',use_dropout=True,keep_prob=0.5,reg_weight=False,reg_lambda=0.0,model_name=sys.argv[2],batch=bat,display=False,noise=False,gradient='adam',learning_rate=0.0000125)


print 'Init: ',ic,' Best:',bc

#mid = auto.get_hidden(data,session=session)

#out = auto.get_output(data,session=session)

#np.save('middle.dat',mid)
#np.save('output.dat',out)


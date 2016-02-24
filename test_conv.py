import tensorflow as tf
from autoencoder import autoencoder
from tools.data_manipulation import batch
from tools.image import display
import numpy as np
import sys 

data = np.loadtxt("conv")

m = np.min(data)

#data = data+abs(m)

units = [96,196,48,24]
act = ['tanh','tanh','tanh']
act2 = ['linear','linear','linear']
auto = autoencoder(units,act)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=False,act=act2)

session = auto.init_network()

bat = batch.rand_batch(data,70)

ic,bc = auto.train(data,batch=bat,n_iters=5000,use_dropout=True,keep_prob=0.5,reg_weight=False,reg_lambda=0.0,model_name='conv_lin_24.mod',display=False,noise=False,gradient='adam',learning_rate=0.000125)


print "Finished. Initial cost: ",ic," Final cost: ",bc

auto.load_model('conv_lin_24.mod',session=session)
#data = data+m
auto.stop_dropout()
red_feat = auto.get_hidden(data,session=session)
out_feat = auto.get_output(data,session=session)
np.savetxt("red_feat_lin_24",red_feat)
np.savetxt("out_feat_lin_24",out_feat)

nd = np.transpose(data)
no = np.transpose(out_feat)
if(sys.argv[1]=='s'):
    for i in range(96):
        #display.save(nd[i],7,7,3,folder='./feat',name='lin_in_24',index=i)
        display.save(no[i],7,7,3,folder='./feat',name='lin_out_24',index=i)
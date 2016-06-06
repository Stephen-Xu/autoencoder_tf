import tensorflow as tf
from autoencoder import autoencoder
#from tools.data_manipulation import batch
from tools.image import display
import numpy as np
import sys 

#data = np.loadtxt("conv64")
data = np.load(sys.argv[1])

data = data.reshape(-1,data.shape[1]*data.shape[2]*data.shape[3])

data = np.transpose(data)

dim = data.shape[1]

id = int(sys.argv[2])

l_r = 0.001
iters = 100000
sav = True
#m = np.min(data)

#data = data+abs(m)

units = [dim,dim*2,id]
act = ['tanh','tanh']
#act = ['linear','tanh','tanh','tanh']
#act2 = ['linear','linear','linear','tanh']
#act2 = ['linear','linear','linear']
auto = autoencoder(units,act)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=False,act=act)

session = auto.init_network()

#bat = batch.rand_batch(data,70)

model = sys.argv[1]+".mod"

ic,bc = auto.train(data,saving=sav,batch=None,n_iters=iters,use_dropout=False,keep_prob=0.8,reg_weight=False,reg_lambda=0.0,model_name=model,display=False,noise=False,gradient='adam',learning_rate=l_r)


print "Finished. Initial cost: ",ic," Final cost: ",bc
print "Initial cost normalized: ",ic/np.linalg.norm(data)**2
print "Best cost normalized: ",bc/np.linalg.norm(data)**2

auto.load_model(model,session=session)
#data = data+m
auto.stop_dropout()


red_feat = auto.get_hidden(data,session=session)
#red_feat = np.reshape(red_feat,[data[1],data[2],data[3],-1])
#red_feat = np.transpose.matrix(red_feat)

out_feat = auto.get_output(data,session=session)
np.save(model+"_red",red_feat)
np.save(model+"_out",out_feat)

Wei = []

for i in range(len(auto.layers)/2-1):
	if i==0:
	        w_0 = session.run(auto.layers[-1-i].W)
		b_0 = session.run(auto.layers[-1-i].b)
		w_1 = session.run(auto.layers[-1-(i+1)].W)
        	b_1 = session.run(auto.layers[-1-(i+1)].b)
        
		Wei = np.matmul(w_1+b_1,w_0+b_0)
	else:
		w = session.run(auto.layers[-1-(i+1)].W)
                b = session.run(auto.layers[-1-(i+1)].b)

		Wei = np.matmul(w+b,Wei)


np.save(model+"_weights",Wei)

'''
nd = np.transpose(data)
no = np.transpose(out_feat)
if(sys.argv[1]=='s'):
    for i in range(64):
        #display.save(nd[i],3,3,3,folder='./feat',name='in_12',index=i)
        display.save(no[i],3,3,3,folder='./feat',name='out_7',index=i)
'''

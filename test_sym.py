import numpy as np
from autoencoder import autoencoder

data = np.random.rand(100,20).astype("float32")

auto = autoencoder([20,15,10],['sigmoid','sigmoid'])


auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=False)

auto.pre_train(data)

auto.train(data,n_iters=20,record_weight=True,reg_weight=False,learning_rate=20.0,reg_lambda=1.0,batch=None,display=False,noise=False)
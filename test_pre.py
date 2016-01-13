from autoencoder import autoencoder
import tensorflow
import numpy as np





auto = autoencoder([3,5,2],['sigmoid','tanh'])

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=False)

data = np.random.rand(20,3).astype("float32");

s = auto.init_network()

print s.run(auto.layers[1].W)

par = auto.pre_train_rbm(data)

print par[2]
print auto.session.run(auto.layers[1].W)


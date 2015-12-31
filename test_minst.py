import tensorflow as tf
import numpy as np
from autoencoder import autoencoder


units = [784,10]
action = ['softplus']

l_rate =  0.01

grad = "gradient"


f = open("../datasets/train-images.idx3-ubyte","r")
arr = np.fromfile(f, '>u1', 60000 * 28 * 28).reshape((60000, 784))
max_value = 0xFF

arr = arr.astype(float)
arr -= max_value / 2.0
arr /= max_value

data = arr

print data.shape

auto = autoencoder(units,action)

auto.generate_encoder()
auto.generate_decoder()


auto.train(data,n_iters=1000,batch=None,display=False,noise=False,gradient=grad,learning_rate=l_rate)


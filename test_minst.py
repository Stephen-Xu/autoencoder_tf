import tensorflow as tf
import numpy as np
from autoencoder import autoencoder


units = [784,10]
action = ['tanh']

l_rate =  0.01

grad = "gradient"


f = open("../datasets/train-images.idx3-ubyte","r")
arr = np.fromfile(f, '>u1', 60000 * 28 * 28).reshape((60000, 784))
max_value = 0xFF

arr = arr.astype(float)
#arr -= max_value / 2.0
#arr /= max_value

data = arr

print data.shape

auto = autoencoder(units,action)

auto.generate_encoder()
auto.generate_decoder()


auto.train(data,n_iters=2000,model_name='model_tahnh.ckpt',batch=None,display=False,noise=False,gradient=grad,learning_rate=l_rate)

auto2 = autoencoder(units,['sofftplus'])

auto2.generate_encoder()
auto2.generate_decoder()

auto2.train(data,n_iters=2000,model_name='model_splus.ckpt',batch=None,display=False,noise=False,gradient=grad,learning_rate=l_rate)


auto3 = autoencoder(units,['sigmoid'])

auto3.generate_encoder()
auto3.generate_decoder()

auto3.train(data,n_iters=2000,model_name='model_sigmo.ckpt',batch=None,display=False,noise=False,gradient=grad,learning_rate=l_rate)

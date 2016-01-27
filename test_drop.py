import tensorflow as tf
import numpy as np
import sys
from autoencoder import autoencoder

data = np.random.rand(10,30).astype("float32")

units =[30,2]
act  = ['tanh']
k = [1.0,0.0001]

auto = autoencoder(units,act)

auto.generate_encoder()
auto.generate_decoder()

session = auto.init_network()

auto.set_dropout(keep_prob=k)
print auto.get_hidden(data,session=session)
print auto.get_output(data,session=session)


'''
data = np.random.rand(1,784).astype("float32")

x = tf.placeholder("float",[1,784])
w = tf.Variable(tf.random_uniform([784,int(sys.argv[2])]))

y = tf.nn.dropout(tf.matmul(x,w),float(sys.argv[1]))



s = tf.Session()

s.run(tf.initialize_all_variables())

m = 0
for i in range(10):
   
    m = m +np.count_nonzero(s.run(y,feed_dict={x:data}))
    print (s.run(y,feed_dict={x:data}).shape)
print 'percentuale media di nulli: ', (m/10.0)/float(sys.argv[2])*100
print 'valore assoluto medio di null: ', m/10.0
'''
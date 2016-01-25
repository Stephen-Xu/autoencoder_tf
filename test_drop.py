import tensorflow as tf
import numpy as np
import sys


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

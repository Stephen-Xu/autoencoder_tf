import tensorflow as tf
import numpy as np


filt = np.loadtxt('./conv').astype("float32")


data = np.random.rand(1,224,224,3).astype("float32")

np.savetxt("data",np.reshape(data,[1*224*224,3]))


x = tf.placeholder("float",[1,224,224,3])
reduced_filters = tf.constant(filt,shape=[7,7,3,96],dtype="float32")
          
conv = tf.nn.conv2d(x,reduced_filters,[1,2,2,1],"VALID")

sess = tf.Session()


sess.run(tf.initialize_all_variables())

result = sess.run(conv,feed_dict={x:data})

np.savetxt("result",np.reshape(result,[1*109*109,96]))

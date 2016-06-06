import tensorflow as tf
import numpy as np
import sys

session = tf.Session()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('conv_height',11,"""Height of kernel filters""")
tf.app.flags.DEFINE_integer('conv_width',11,"""Width of kernel filters""")
tf.app.flags.DEFINE_integer('channels',3,"""Number of image channels""")
tf.app.flags.DEFINE_integer('iters',100000,"""Iterations number""")
tf.app.flags.DEFINE_float('learning_rate',0.1,"""Learning rate for optimizer""")
tf.app.flags.DEFINE_integer('batch',100,"""Size of batches.""")
tf.app.flags.DEFINE_integer('heigth',224,"""Height of input image""")
tf.app.flags.DEFINE_integer('width',224,"""Width of input image""")


model_name = sys.argv[1]



#W = np.load("levels_alex/"+model_name+".mod_weights.npy")

padding = [1,1,1,1]

ori = np.load("levels_alex/"+model_name)
#ori = np.reshape(np.transpose(ori),[256,192,3,3])

ori = np.transpose(ori)
#ori = np.load("levels/"+model_name)
#ori = np.matrix.transpose(ori,[2,3,1,0])
n_ori_filters = ori.shape[-1]
n_channel = ori.shape[-2]
n_k_filters = ori.shape[0]

red = np.load("levels_alex/"+model_name+".mod_red.npy")
#red = np.reshape(red,[ori.shape[0],ori.shape[1],ori.shape[2],-1])
red = np.transpose(np.reshape(np.transpose(red),[-1,n_channel,n_k_filters,n_k_filters]))
#red = np.loadtxt("r0.txt")
#red = np.reshape(red,[11,11,3,-1])

n_red_filters = red.shape[-1]


W = np.random.normal(size=[n_red_filters,n_ori_filters]).astype("float32")
W2 = np.random.normal(size=[n_red_filters,n_red_filters]).astype("float32")


original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")

reconstruction_filters = tf.Variable(W,name="W",dtype="float32")
rec2 = tf.Variable(W2,name="W2",dtype="float32")
#reconstruction_filters = tf.expand_dims(tf.expand_dims(reconstruction_filters,0),0)

x = tf.placeholder("float",[None,ori.shape[0],ori.shape[1],ori.shape[2]])

conv_original = tf.nn.conv2d(x,original_filters,padding,"VALID")
conv_reduced = tf.nn.conv2d(x,reduced_filters,padding,"VALID")

#conv_reconstruction = tf.nn.conv2d(conv_reduced,reconstruction_filters,padding,"VALID")

resh = tf.reshape(conv_reduced,[-1,n_red_filters])

conv_reconstruction = tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(resh,rec2)),reconstruction_filters))

hat_c = tf.reshape(conv_reconstruction,[FLAGS.batch,n_ori_filters])
ori_c = tf.reshape(conv_original,[FLAGS.batch,n_ori_filters])
loss = tf.reduce_mean(tf.pow(ori_c-hat_c,2))

tr = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

session.run(tf.initialize_all_variables())

actual_batch = np.random.uniform(-250.0,250.0,[FLAGS.batch,ori.shape[0],ori.shape[1],ori.shape[2]])

initial_cost = session.run(loss,feed_dict={x:actual_batch})
cost = initial_cost
for i in range(FLAGS.iters):
    actual_batch = np.random.uniform(-250.0,250.0,[FLAGS.batch,ori.shape[0],ori.shape[1],ori.shape[2]])

    _, c = session.run([tr,loss],feed_dict={x:actual_batch})
    print "Cost at iter ",i," : ",c
    if(c<cost):
	cost = c
        val = session.run(reconstruction_filters)
 	np.save("recons/alex_expl"+model_name,np.squeeze(val))
        

actual_batch = np.random.uniform(-250.0,250.0,[FLAGS.batch,ori.shape[0],ori.shape[1],ori.shape[2]])
final_cost = session.run(loss,feed_dict={x:actual_batch})



print "Initial cost: ",initial_cost," Final cost: ",final_cost," Best: ",cost
print "ori: ", session.run(conv_original,feed_dict={x:actual_batch})
print "red: ", session.run(conv_reconstruction,feed_dict={x:actual_batch})








import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join

session = tf.Session()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('path','/home/ceru/datasets/ILSVRC2012_VAL_SET/old/pre_images/',"""Data folder""")
tf.app.flags.DEFINE_integer('conv_height',3,"""Height of kernel filters""")
tf.app.flags.DEFINE_integer('conv_width',3,"""Width of kernel filters""")
tf.app.flags.DEFINE_integer('channels',3,"""Number of image channels""")
tf.app.flags.DEFINE_integer('iters',600000,"""Iterations number""")
tf.app.flags.DEFINE_float('learning_rate',0.01,"""Learning rate for optimizer""")
tf.app.flags.DEFINE_integer('batch',100,"""Size of batches.""")
tf.app.flags.DEFINE_integer('heigth',224,"""Height of input image""")
tf.app.flags.DEFINE_integer('width',224,"""Width of input image""")


W = np.load("levels/conv1_2_0.npy.mod_weights.npy")

padding = [1,1,1,1]
ori1 = np.load("levels/conv1_1_0.npy")
bias1 = np.load("levels/conv1_1_1.npy")
ori2 = np.load("levels/conv1_2_0.npy")


ori1_filters_number = ori1.shape[0]
ori2_filters_number = ori2.shape[0]

red = np.load("levels/conv1_2_0.npy.mod_red.npy")
red = np.reshape(red,[3,3,64,7])

ori1 = np.matrix.transpose(ori1,[2,3,1,0])
ori2 = np.matrix.transpose(ori2,[2,3,1,0])


'''ori = np.reshape(ori,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,ori_filters_number])
red = np.loadtxt("./red_7").astype("float32")
#red = np.loadtxt("./red_32").astype("float32")
red_filters_number = red.shape[1]
red = np.reshape(red,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,red_filters_number])
'''

original_filters_1 = tf.constant(ori1,shape=ori1.shape,dtype="float32")
original_filters_2 = tf.constant(ori2,shape=ori2.shape,dtype="float32")
reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
##
#layer_1 = tf.Variable(tf.random_normal([red_filters_number,red_filters_number]),name="l_w_1",dtype="float32")
##
#reconstruction_filters = tf.Variable(tf.random_normal([red_filters_number,ori_filters_number]),name="W",dtype="float32")
reconstruction_filters = tf.Variable(W,name="W",dtype="float32")

reconstruction_filters = tf.expand_dims(tf.expand_dims(reconstruction_filters,0),0)


#x = tf.placeholder("float",[None,FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
x = tf.placeholder("float",[None,FLAGS.conv_width*3,FLAGS.conv_width*3,FLAGS.channels])

#W_due = tf.Variable(tf.random_normal([red_filters_number,red_filters_number]),name="W_due",dtype="float32")

#rec_filters_due = tf.expand_dims(tf.expand_dims(W_due,0),0)

conv_original_1 = tf.nn.conv2d(x,original_filters_1,padding,"VALID")
relu_1 = tf.nn.relu(conv_original_1)

#############

conv_original_2 = tf.nn.conv2d(relu_1,original_filters_2,padding,"VALID")

conv_reduced = tf.nn.conv2d(relu_1,reduced_filters,padding,"VALID")


conv_reconstruction = tf.nn.conv2d(conv_reduced,reconstruction_filters,padding,"VALID")
#conv_reconstruction = tf.nn.conv2d(conv_reduced_proc,reconstruction_filters,padding,"VALID")


files = [FLAGS.path+f for f in listdir(FLAGS.path) if isfile(join(FLAGS.path, f))]

#hat_c = tf.reshape(conv_reconstruction,[FLAGS.batch,ori2_filters_number])
hat_c = tf.reshape(conv_reconstruction,[FLAGS.batch*5*5,ori2_filters_number])
#ori_c = tf.reshape(conv_original_2,[FLAGS.batch,ori2_filters_number])
ori_c = tf.reshape(conv_original_2,[FLAGS.batch*5*5,ori2_filters_number])
loss = tf.reduce_sum(tf.pow(ori_c-hat_c,2))

tr = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

file_queue = tf.train.string_input_producer(files, shuffle=True, capacity=len(files))
reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

image = tf.image.decode_jpeg(value,channels=3)

image = tf.to_float(image)
image.set_shape([FLAGS.heigth,FLAGS.width,FLAGS.channels])
image = tf.random_crop(image,[FLAGS.conv_width*3,FLAGS.conv_width*3,FLAGS.channels])
image = tf.expand_dims(image,0)

get_batch = tf.train.batch([image], batch_size=FLAGS.batch, num_threads=2016, capacity=200, enqueue_many=True)

session.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=session)



actual_batch = session.run(get_batch)

############

#print session.run(relu1,feed_dict={x:acutal_batch}).shape
##########



initial_cost = session.run(loss,feed_dict={x:actual_batch})
cost = initial_cost
for i in range(FLAGS.iters):
    actual_batch = session.run(get_batch)
    _, c = session.run([tr,loss],feed_dict={x:actual_batch})
    print "Cost at iter ",i," : ",c
    if(c<cost):
	cost = c
        val = session.run(reconstruction_filters)
 	np.save("reconstruction_conv_1_2",np.squeeze(val))
        
actual_batch = session.run(get_batch)
final_cost = session.run(loss,feed_dict={x:actual_batch})



print "Initial cost: ",initial_cost," Final cost: ",final_cost," Best: ",cost
print "ori: ", session.run(conv_original,feed_dict={x:actual_batch})
print "red: ", session.run(conv_reconstruction,feed_dict={x:actual_batch})








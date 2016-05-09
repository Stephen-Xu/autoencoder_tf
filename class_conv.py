import tensorflow as tf
import numpy as np


session = tf.Session()

tf.app.flags.DEFINE_string('path','/home/ceru/datasets/ILSVRC2012_VAL_SET/pre_images/',"""Data folder""")
tf.app.flags.DEFINE_integer('conv_height',3,"""Height of kernel filters""")
tf.app.flags.DEFINE_integer('conv_width',3,"""Width of kernel filters""")
tf.app.flags.DEFINE_integer('channels',3,"""Number of image channels""")
tf.app.flags.DEFINE_integer('iters',1000,"""Iterations number""")
tf.app.flags.DEFINE_float('learning_rate',0.00125,"""Learning rate for optimizer""")
tf.app.flags.DEFINE_integer('batch',100,"""Size of batches.""")


ori = np.loadtxt("./conv64").astype("float32")
ori_filters_number = ori.shape[1]
ori = np.reshape(ori,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,ori_filters_number])
red = np.loadtxt("./red_7").astype("float32")
red_filters_number = red.shape[1]
red = np.reshape(red,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,red_filters_number])


original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
reconstruction_filters = tf.Variable(W,tf.random_normal([red.shape,ori.shape]),dtype="float32")


x = tf.placeholder("float",[None,FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])


conv_original = tf.nn.conv2d(x,original_filters,padding,"VALID")
conv_reduced = tf.nn.conv2d(x,reduced_filters,padding,"VALID")
conv_reconstruction = tf.nn.conv2d(conv_reduced,reconstruciotn_filters,padding,"VALID")


files = [FLAGS.path+f for f in listdir(FLAGS.path) if isfile(join(FLAGS.path, f))]

hat_c = self.output(tf.reshape(conv_reconstruction,[FLAGS.batch,ori_filters_number]))
ori_c = tf.reshape(conv_original,[FLAGS.batch,ori_filters_number])
loss = tf.reduce_mean(tf.pow(ori_c-hat_c,2))

tr = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

file_queue = tf.train.string_input_producer(files, shuffle=True, capacity=len(files))
reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

image = tf.image.decode_jpeg(value,channels=3)

image = tf.to_float(image)
image.set_shape([FLAGS.heigth,FLAGS.width,FLAGS.channels])
image = tf.random_crop(image,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
image = tf.expand_dims(image,0)

get_batch = tf.train.batch([image], batch_size=FLAGS.batch, num_threads=2016, capacitapacity=200, enqueue_many=True)

session.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=self.session)


actual_batch = session.run(get_batch)

initial_cost = session.run(loss,feed_dict={x:actual_batch})
cost = initial_cost
for i in range(FLAGS.iters):
    actual_batch = session.run(get_batch)
    _, c = session.run([tr,loss],feed_dict={x:actual_batch})
    print "Cost at iter ",i," : ",c
    if(c<cost):
        np.save("rec_conv",session.run(W))

actual_batch = session.run(get_batch)
final_cost = session.run(loss,feed_dict={x:actual_batch})



print "Initial cost: ",initial_cost," Final cost: ",final_cost," Best: ",c









import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from tools.image import display


original_file = './conv'


ori = np.loadtxt(original_file).astype("float32")
ori = np.reshape(ori,[7,7,3,96])




mypath = './imgs/'
files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
    
file_queue = tf.train.string_input_producer(files, shuffle=True, capacity=len(files))
reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

image = tf.image.decode_jpeg(value)
image.set_shape([179,179,3])
image = tf.image.resize_images(image,224,224)
image.set_shape([224,224,3])

image = tf.image.convert_image_dtype(image,dtype=tf.float32)
image = tf.random_crop(image,[7,7,3])
image = tf.expand_dims(image,[0])


batch = 10
sq = tf.train.batch([image], batch_size=batch, num_threads=7, capacity=200, enqueue_many=True)


val = tf.placeholder("float",[batch,7,7,3])

session = tf.Session()
session.run(tf.initialize_all_variables())

q = tf.nn.conv2d(val,ori,[1,2,2,1],"VALID")

threads = tf.train.start_queue_runners(sess=session)




res = session.run(q,feed_dict={val:session.run(sq)})

print res.shape
print res[0,:,:,:]

'''
print session.run(sq).shape
print np.mean(session.run(sq))

for i in range(10):
    res = session.run(q,feed_dict={val:session.run(sq)})
    print np.mean(res)
    print res.shape
'''
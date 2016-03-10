import tensorflow as tf
import numpy as np
from tools.data_manipulation import image_batch


mypath = '../datasets/ILSVRC2012_VAL_SET/pre_images/'
original_file = './conv'
reduced_file = './red_feat_lin_24'



ori = np.loadtxt(original_file).astype("float32")
ori = np.reshape(ori,[7,7,3,96])
red = np.loadtxt(reduced_file).astype("float32")
red = np.reshape(red,[7,7,3,24])

image = tf.placeholder(dtype="float32",shape=[None,None,None,None])

orig_bank = tf.constant(ori,shape=ori.shape,dtype="float32")
red_bank = tf.constant(red,shape=red.shape,dtype="float32")




 

 
c_orig = tf.nn.conv2d(image,orig_bank,[1,2,2,1],"VALID")
c_red = tf.nn.conv2d(image,red_bank,[1,2,2,1],"VALID")
b = image_batch.image_batch(mypath)

session = tf.Session()

session.run(tf.initialize_all_variables())
#tf.train.start_queue_runners(sess=session)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)

print "ok1"


inputs = b[0]
try:
    while not coord.should_stop():
        # Run training steps or whatever
        print session.run(inputs)
        print b[1]

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.

session.close()


'''
#c = session.run(c_orig,feed_dict={image:image_batch.image_batch(mypath,batch_size=20)})

print b
c = session.run(b[0])
print ok
print c
#print session.run(b[0])

#print np.mean(abs(session.run(c_orig,feed_dict={image:session.run(b[0])})))
'''




'''
for i in range(4):
    print "ok"
    
    print np.mean(abs(session.run(c_orig,feed_dict={image:session.run(b[i])})))
    print np.mean(abs(session.run(c_red,feed_dict={image:session.run(b[i])})))


'''

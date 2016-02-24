import tensorflow as tf
import sys
import numpy as np
from scipy import misc
from tools.image import display

orig_bank = np.loadtxt("conv")
mean_image = np.loadtxt("./mean_image").astype("uint8")
mean_image = np.reshape(mean_image,[224,224,3])
#mean_image = misc.imread('./mean_image.jpeg').astype("float32")

f_b = np.loadtxt("./red_feat")
w_1 = np.loadtxt("./weights1")
w_2 = np.loadtxt("./weights2")
w_3 = np.loadtxt("./weights3")
w_4 = np.loadtxt("./weights4")


file_queue = tf.train.string_input_producer([sys.argv[1]])

reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

image = tf.image.decode_jpeg(value)

image.set_shape([400, 400, 3])

m_i = tf.constant(mean_image,shape=mean_image.shape,dtype=tf.float32)

data = tf.image.resize_images(image, 224, 224)
#data = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
#data = tf.image.convert_image_dtype(data,dtype=tf.float32)
data = tf.image.convert_image_dtype(data,dtype=tf.float32)
#data  = data-m_i



data = tf.expand_dims(data, 0)


f_b = np.reshape(f_b,[7,7,3,10])
f_o = np.reshape(orig_bank,[7,7,3,96])
f_bank = tf.constant(f_b,dtype=tf.float32)
f_o_bank = tf.constant(f_o,dtype=tf.float32)
W1 = tf.constant(w_1,shape=w_1.shape,dtype=tf.float32)
W2 = tf.constant(w_2,shape=w_2.shape,dtype=tf.float32)
W3 = tf.constant(w_3,shape=w_3.shape,dtype=tf.float32)
W4 = tf.constant(w_4,shape=w_4.shape,dtype=tf.float32)

conv = tf.nn.conv2d(data, f_bank, [1, 2, 2, 1],"VALID")
orig_conv = tf.nn.conv2d(data,f_o_bank,[1,2,2,1],"VALID")
c_re = tf.reshape(conv,[109*109,10])

l1 = tf.tanh(tf.matmul(c_re,W1))
l2 = tf.tanh(tf.matmul(l1,W2))
l3 = tf.tanh(tf.matmul(l2,W3))
l4 = tf.tanh(tf.matmul(l3,W4))

session = tf.Session()

session.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=session)

sq = tf.squeeze(orig_conv,[0])

final = tf.reshape(sq,[109*109,96])

res = session.run(final)

print res.shape
'''display.display(res,224,224,3)
display.display((res-mean_image),224,224,3)

dog_image = np.loadtxt("./dog_mean").astype("float32")
dog_image = np.reshape(dog_image,[224,224,3])
display.display(dog_image,224,224,3)

print dog_image[:,0]
print (res-mean_image)[0]'''

#np.savetxt("orig_conv_out_res",res)
matlab = np.loadtxt("net_conv_out")

print np.mean(np.mean(np.abs(res-matlab)))

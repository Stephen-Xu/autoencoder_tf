import tensorflow as tf
import sys
import numpy as np
from tools.image import display
import scipy.io

mat_bank = scipy.io.loadmat("filters.mat")
orig_bank = mat_bank['w']


file_queue = tf.train.string_input_producer([sys.argv[1]])

reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

image = tf.image.decode_jpeg(value)

image.set_shape([400, 400, 3])

'''-----'''

data = tf.image.resize_images(image, 224, 224)
#data = tf.ceil(data)
#data = tf.image.convert_image_dtype(data,dtype=tf.uint8)

#data = tf.image.flip_up_down(data)


mat_img = scipy.io.loadmat("dog.mat")
noise_img = mat_img['img']

data = tf.constant(noise_img,dtype=tf.float32)

data = tf.expand_dims(data, 0)


f_b = np.loadtxt("./red_feat_lin")
w_1 = np.loadtxt("./weights1_lin")
b_1 = np.loadtxt("./b1_lin")
w_2 = np.loadtxt("./weights2_lin")
b_2 = np.loadtxt("./b2_lin")
w_3 = np.loadtxt("./weights3_lin")
b_3 = np.loadtxt("./b3_lin")
w_4 = np.loadtxt("./weights4_lin")
b_4 = np.loadtxt("./b4_lin")
W1 = tf.constant(w_1,shape=w_1.shape,dtype=tf.float32)
B1 = tf.constant(b_1,shape=b_1.shape,dtype=tf.float32)
W2 = tf.constant(w_2,shape=w_2.shape,dtype=tf.float32)
B2 = tf.constant(b_2,shape=b_2.shape,dtype=tf.float32)
W3 = tf.constant(w_3,shape=w_3.shape,dtype=tf.float32)
B3 = tf.constant(b_3,shape=b_3.shape,dtype=tf.float32)
W4 = tf.constant(w_4,shape=w_4.shape,dtype=tf.float32)
B4 = tf.constant(b_4,shape=b_4.shape,dtype=tf.float32)
f_b = np.reshape(f_b,[7,7,3,10])
f_bank = tf.constant(f_b,dtype=tf.float32)
conv = tf.nn.conv2d(data, f_bank, [1, 2, 2, 1],"VALID")
c_re = tf.reshape(conv,[109*109,10])

l1 = (tf.matmul(c_re,W1)+B1)
l2 = (tf.matmul(l1,W2)+B2)
l3 = (tf.matmul(l2,W3)+B3)
l4 = tf.matmul(l3,W4)+B4
f_b = np.reshape(f_b,[7,7,3,10])

f_o_bank = tf.constant(orig_bank,dtype=tf.float32)


orig_conv = tf.nn.conv2d(data,f_o_bank,[1,2,2,1],"VALID")
#t_conv = tf.nn.conv2d_transpose(data,f_o_bank,[1,2,2,1],"VALID")
sq = tf.squeeze(orig_conv,[0])
sql = tf.reshape(l4,[109,109,96])
final = tf.reshape(sq,[109,109,96])





session = tf.Session()

session.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=session)

orig = session.run(sq)
red = session.run(sql)
#red = session.run(l4)
#print red.shape
print np.sum(abs(orig-red))
print orig[0,1:6,1:6]
print red[0,1:6,1:6]
#mat_out = scipy.io.loadmat("out_conv.mat")
#out_conv = mat_out['res']


#f = session.run(f_o_bank)

#print np.sum(abs(f-orig_bank))
'''
print np.sum(abs(res-out_conv))

print res.shape


print res[0,0,0]
print out_conv[0,0,0]
print res[1,2,0]
print out_conv[1,2,0]
print res[32,21,0]
print out_conv[32,21,0]

for i in range(20):
    print 'r: ',res[20,i,20]
    print 'm: ',out_conv[20,i,20]
    print 'r2: ',res[i,20,10]
    print 'm2: ',out_conv[i,20,10]



print '---------------'
print res
print '---------------'
print out_conv
a={}
a['res']=res
scipy.io.savemat('pyt_res.mat',a)
'''
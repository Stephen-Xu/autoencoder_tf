from classifier import classifier
import tensorflow as tf
import numpy as np
import scipy.io as sio


def get_conv(data):
    ori = np.loadtxt('./conv').astype("float32")
    ori_filters_number = ori.shape[1]
    ori = np.reshape(ori,[7,7,3,ori_filters_number])
    red = np.loadtxt('./red_feat_lin_24').astype("float32")
    red_filters_number = red.shape[1]
    red = np.reshape(red,[7,7,3,red_filters_number])
    
    with tf.Graph().as_default():
        
        
        x = tf.placeholder("float",[None,224,224,3])###immagini
        
        
        original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
        reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
        
        conv_reduced = tf.nn.conv2d(x,reduced_filters,[1,2,2,1],"VALID")
        conv_original = tf.nn.conv2d(x,original_filters,[1,2,2,1],"VALID")
    
        
        l_session = tf.Session()
        
        l_session.run(tf.initialize_all_variables())
        
        o = l_session.run(conv_reduced,feed_dict={x:data})
        
        r = l_session.run(conv_original,feed_dict={x:data})
        
    
    return o,r



mat = sio.loadmat("./single_ex.mat")
data = mat['a']

data_ = np.expand_dims(data,0)


units = [24,192,96]   #################
act = ['tanh','linear']

cl = classifier(units,act)


cl.generate_classifier()


cl.init_network()


cl.load_model(session=cl.session,name='./converted.mdl')


o,r = get_conv(data_)



print o.shape,np.mean(o)
print r.shape,np.mean(r)


l_o = np.squeeze(o)

l_o = np.reshape(l_o,[109*109,24])

c = cl.session.run(cl.output(l_o))

print c.shape

c_o = np.reshape(c,[109,109,96])

c_o = np.expand_dims(c_o,0)

print np.mean(pow(c_o-r,2))
print c_o.shape,np.mean(c_o)
print c_o[0,0:2,0:2,0]
print r[0,0:2,0:2,0]

for i in range(len(cl.layers)):
    w = cl.session.run(cl.layers[i].W)
    print "W ",i,":",np.mean(w),",",np.std(w)
    
    

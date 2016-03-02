import tensorflow as tf
import layer
from os import listdir
from os.path import isfile, join
import numpy as np


#FLAAAAAAAAAAAGS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('iters',2000,"""Number of iterations.""")
tf.app.flags.DEFINE_string('model','./converted.mdl',"""File for saving model.""")
tf.app.flags.DEFINE_integer('batch',100,"""Size of batches.""")
tf.app.flags.DEFINE_integer('heigth',224,"""Height of images""")
tf.app.flags.DEFINE_integer('width',224,"""Width of images""")
tf.app.flags.DEFINE_string('path','/home/ceru/datasets/ILSVRC2012_VAL_SET/pre_images/',"""Data folder""")
tf.app.flags.DEFINE_string('original','./conv',"""File for original filters""")
tf.app.flags.DEFINE_string('reduced','./red_feat_lin_24',"""File for reduced filters""")
tf.app.flags.DEFINE_integer('conv_width',7,"""Convolutional width""")
tf.app.flags.DEFINE_integer('channels',3,"""Number of images channel""")
tf.app.flags.DEFINE_integer('out_conv_dim',109,"""Shape of convolutional output""")


class classifier(object):
    def __init__(self,units,act):
        assert len(units)-1==len(act),"Number of layers and number of activation functions must be the same (len(units)-1 == len(act_func))"
        self.layers = []
        self.units = units
        self.act_func = act
        self.use_euristic=False
        self.initialized = False
        self.session = None
        self.use_droput = False
        self.keep_prob_dropout = 0.5
        
        
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_value, traceback):
        pass
    
    
    def generate_classifier(self,euris=False,mean_w=0.0,std_w=1.0):
        for i in range(len(self.units)-1):
            self.layers.append(layer.layer([self.units[i],self.units[i+1]],activation=self.act_func[i],mean=mean_w,std=std_w,eur=euris,))
        if(euris):
            self.use_euristic = True
            
    
    def init_network(self,list_var=None):
       
        if(list_var is None):
            init = tf.initialize_all_variables()
        else:
            init = tf.initialize_variables(list_var)
        
       
        sess = tf.Session()
    
        sess.run(init)
        
        self.initialized=True
        self.session = sess
        
        
        return sess
    
    
    
            
    def load_model(self,name,session=None,saver=None):
            if(saver is None):
                saver = tf.train.Saver()
            if(session is None):
                init = tf.initialize_all_variables()
                session = tf.Session()
                session.run(init) 
            saver.restore(session, name)
        
        
    def output(self,x,lev=None):
        
        assert len(self.layers)!=0, 'You must generate the classifier in order to get an output!'
        if lev is None:
            lev = 0
        if(lev==len(self.act_func)-1):
            if(self.use_droput):
                if(not isinstance(self.keep_prob_dropout,list)):
                    return self.layers[lev].output_dropout(x,keep_prob=self.keep_prob_dropout)
                else:
                
                    return self.layers[lev].output_dropout(x,keep_prob=float(self.keep_prob_dropout[lev]))
            else:
                return self.layers[lev].output(x)
        else:
            return self.output(self.layers[lev].output(x),lev+1)   

    
        
    def train(self):#FLAAAGSSS!:
        
        if(self.session is None):
            session = self.init_network()
        elif(self.session is None):
            self.session = session
        
       # self.use_droput=use_dropout
       # self.keep_prob_dropout=keep_prob
        
    
        files = [FLAGS.path+f for f in listdir(FLAGS.path) if isfile(join(FLAGS.path, f))]
        
        ori = np.loadtxt(FLAGS.original).astype("float32")
        ori_filters_number = ori.shape[1]
        ori = np.reshape(ori,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,ori_filters_number])
        red = np.loadtxt(FLAGS.reduced).astype("float32")
        red_filters_number = red.shape[1]
        red = np.reshape(red,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,red_filters_number])



        original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
        reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
        
        
        x = tf.placeholder("float",[FLAGS.batch,FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])###immagini
        
        conv_reduced = tf.nn.conv2d(x,reduced_filters,[1,1,1,1],"VALID")
        conv_original = tf.nn.conv2d(x,original_filters,[1,1,1,1],"VALID")
        '''hat_c = self.output(tf.reshape(conv_reduced,[FLAGS.batch,red_filters_number]))
        loss = tf.reduce_mean((tf.pow(tf.reshape(conv_original,[FLAGS.batch,ori_filters_number])-hat_c,2)))
        
        tr = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        '''
        file_queue = tf.train.string_input_producer(files, shuffle=True, capacity=len(files))
        reader = tf.WholeFileReader()
        key,value = reader.read(file_queue)

        image = tf.image.decode_jpeg(value)
        image.set_shape([FLAGS.heigth,FLAGS.width,FLAGS.channels])

        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
        image = tf.random_crop(image,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
        image.set_shape([FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
        image = tf.expand_dims(image,[0])


        get_batch = tf.train.batch([image], batch_size=FLAGS.batch, num_threads=7, capacity=200, enqueue_many=True)
        
            
        
        self.session.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=self.session)       
        
        saver = tf.train.Saver()
        
       
        
        #initial_cost = self.session.run(loss)
        
        #for i in range(FLAGS.iters):
        for i in range(6): 
            actual_batch = self.session.run(get_batch)
            print actual_batch.shape
            print self.session.run(conv_reduced).shape
            #_, c = self.session.run([tr,loss],feed_dict={x:actual_batch})
            #print c
            
        #final_cost = self.session.run(loss)
        
        
        print "initial cost: ",initial_cost," Final cost: ",final_cost
                  
        saver.save(self.session,FLAGS.model)
        
        
       
        
        
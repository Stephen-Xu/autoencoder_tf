import tensorflow as tf
import layer
from os import listdir
from os.path import isfile, join
import numpy as np


#FLAAAAAAAAAAAGS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('iters',15000,"""Number of iterations.""")
tf.app.flags.DEFINE_string('model','./converted.mdl',"""File for saving model.""")
tf.app.flags.DEFINE_integer('batch',100,"""Size of batches.""")
tf.app.flags.DEFINE_integer('heigth',224,"""Height of images""")
tf.app.flags.DEFINE_integer('width',224,"""Width of images""")
tf.app.flags.DEFINE_string('path','/home/ceru/datasets/ILSVRC2012_VAL_SET/pre_images/',"""Data folder""")
tf.app.flags.DEFINE_string('original','./conv64',"""File for original filters""")
tf.app.flags.DEFINE_string('reduced','./red_7',"""File for reduced filters""")
tf.app.flags.DEFINE_integer('conv_width',3,"""Convolutional width""")
tf.app.flags.DEFINE_integer('channels',3,"""Number of images channel""")
tf.app.flags.DEFINE_integer('out_conv_dim',1,"""Shape of convolutional output""")
tf.app.flags.DEFINE_float('learning_rate',0.00125,"""Learning rate for optimizer""")
tf.app.flags.DEFINE_float('reg_weight',200.0,"""Regularization paramter""")


class classifier(object):
    def __init__(self,units,act):
        assert len(units)-1==len(act),"Number of layers and number of activation functions must be the same (len(units)-1 == len(act_func))"
        self.layers = []
        self.units = units
        self.act_func = act
        self.use_euristic=False
        self.initialized = False
        self.session = None
        self.use_dropout = False
        self.keep_prob_dropout = 0.5
        self.generated = False
    
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_value, traceback):
        pass
    
    
    def generate_classifier(self,euris=False,mean_w=0.0,std_w=1.0,dropout=False,keep_prob_dropout=0.5):
        for i in range(len(self.units)-1):
            self.layers.append(layer.layer([self.units[i],self.units[i+1]],activation=self.act_func[i],mean=mean_w,std=std_w,eur=euris))
        if(euris):
            self.use_euristic = True
            
        self.use_dropout = dropout
        self.keep_prob_dropout = keep_prob_dropout
    
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
    
    def test_model(self,data,session=None,model=None):
        
        if(self.session is None):
            init = tf.initialize_alla_variables()
            session = tf.Session()
            session.run(init)
            self.session = session
            
        if(not model is None):
            self.load_model(model,session=self.session)
            
        #ori = np.loadtxt(FLAGS.original).astype("float32")
        ori = np.loadtxt("./conv64").astype("float32")
        ori_filters_number = ori.shape[1]
        ori = np.reshape(ori,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,ori_filters_number])
        #red = np.loadtxt(FLAGS.reduced).astype("float32")
        red = np.loadtxt("./red_7").astype("float32")
        red_filters_number = red.shape[1]
        red = np.reshape(red,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,red_filters_number])

    #######################################MODIFICARE!!!!!!!!!!!!!!!!!
        
        temp = tf.constant(data,shape=data.shape,dtype="float32")
        patch =  tf.random_crop(temp,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
        patch = tf.expand_dims(patch,[0])
    #######################################MODIFICARE!!!!!!!!!!!!!!!!!

        original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
        reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
          
        conv_reduced = tf.nn.conv2d(patch,reduced_filters,[1,1,1,1],"VALID")
        conv_original = tf.nn.conv2d(patch,original_filters,[1,1,1,1],"VALID")   
        
        out = self.output(tf.reshape(conv_reduced,[1,red_filters_number]))
    
        
        return session.run(out),session.run(conv_original)
        
    
            
    def load_model(self,name,session=None,saver=None):
            if(saver is None):
                saver = tf.train.Saver()
            if(session is None):
                init = tf.initialize_all_variables()
                #session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                session = tf.Session()
                session.run(init) 
            saver.restore(session, name)
        
        
    def output(self,x,lev=None):
        assert len(self.layers)!=0, 'You must generate the classifier in order to get an output!'
        if lev is None:
            lev = 0
        if(lev==len(self.act_func)-1):
            if(self.use_dropout):
		
                if(not isinstance(self.keep_prob_dropout,list)):
                    return self.layers[lev].output_dropout(x,keep_prob=self.keep_prob_dropout)
                else:
                
                    return self.layers[lev].output_dropout(x,keep_prob=float(self.keep_prob_dropout[lev]))
            else:
                return self.layers[lev].output(x)
        else:
            return self.output(self.layers[lev].output(x),lev+1)   



    def get_convolution(self,x,padding=None):
        
        if(padding is None):
            padding = [1,1,1,1]
        
        #ori = np.loadtxt(FLAGS.original).astype("float32")
        ori = np.loadtxt("./conv64").astype("float32")
        ori_filters_number = ori.shape[1]
        ori = np.reshape(ori,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,ori_filters_number])
        #red = np.loadtxt(FLAGS.reduced).astype("float32")
        red = np.loadtxt("./red_7").astype("float32")
        red_filters_number = red.shape[1]
        red = np.reshape(red,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels,red_filters_number])



        original_filters = tf.constant(ori,shape=ori.shape,dtype="float32")
        reduced_filters = tf.constant(red,shape=red.shape,dtype="float32")
        
        conv_reduced = tf.nn.conv2d(x,reduced_filters,padding,"VALID")
        conv_original = tf.nn.conv2d(x,original_filters,padding,"VALID")        
        
        return conv_reduced,conv_original,ori_filters_number,red_filters_number

    def stop_dropout(self):
        self.use_dropout=False
        
    def train(self,session=None):#FLAAAGSSS!:
        
        
        with tf.Graph().as_default():
            
        #######################
            if(not(session is None)):
                self.session = session
            elif(self.session is None):
                self.session = self.init_network()
            
            if(not(self.generated)):
                self.generate_classifier()
                #self.generate_classifier(euris=True,dropout=True,keep_prob_dropout=[1.0,1.0,0.5])
                self.generate_classifier(euris=True,dropout=False)
                self.generated = True    
        
        
        
       # self.use_droput=use_dropout
       # self.keep_prob_dropout=keep_prob
        
    
            files = [FLAGS.path+f for f in listdir(FLAGS.path) if isfile(join(FLAGS.path, f))]
        
        
        
            x = tf.placeholder("float",[None,FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])###immagini
            #x = tf.placeholder("float",[None,None,None,FLAGS.channels])###immagini
           
           
            conv_reduced, conv_original,ori_filters_number,red_filters_number = self.get_convolution(x)           
            hat_c = self.output(tf.reshape(conv_reduced,[FLAGS.batch,red_filters_number]))
            ori_c = tf.reshape(conv_original,[FLAGS.batch,ori_filters_number])
            ori_1 = tf.reshape(conv_original,[1,ori_filters_number])
            hat_1 = self.output(tf.reshape(conv_reduced,[1,red_filters_number]))
            loss = tf.reduce_mean(tf.pow(ori_c-hat_c,2))

         
            
            
            for l in range(len(self.layers)):
                if(l==0):
                    c_w = tf.pow(tf.reduce_sum(tf.pow((self.layers[l].W),2)),0.5)/((self.layers[l].n_out+self.layers[l].n_in)**0.5)
                else:
                    c_w = c_w+tf.pow(tf.reduce_sum(tf.pow((self.layers[l].W),2)),0.5)/((self.layers[l].n_out+self.layers[l].n_in)**0.5)
                
               
        
            reg_loss = loss+FLAGS.reg_weight*c_w
            tr = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
            tr_l  = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(reg_loss)        

            file_queue = tf.train.string_input_producer(files, shuffle=True, capacity=len(files))
            reader = tf.WholeFileReader()
            key,value = reader.read(file_queue)

            image = tf.image.decode_jpeg(value,channels=3)
        
    	    #import scipy.io as sio	
            #image = tf.image.convert_image_dtype(image,dtype=tf.float32)
            image = tf.to_float(image)
            #a_image = sio.loadmat(FLAGS.mean_image)
	    #m_image = a_image['mean_image'].astype("float32")
	    #m_image = np.expand_dims(m_image,0)
            image.set_shape([FLAGS.heigth,FLAGS.width,FLAGS.channels])
            #image = image-m_image
            image = tf.random_crop(image,[FLAGS.conv_width,FLAGS.conv_width,FLAGS.channels])
      
            #image = tf.expand_dims(image,[0])


	   
	    

            get_batch = tf.train.batch([image], batch_size=FLAGS.batch, num_threads=2016, capacity=200, enqueue_many=True)
        
        
        
            self.session.run(tf.initialize_all_variables())
            tf.train.start_queue_runners(sess=self.session)       
            
            saver = tf.train.Saver()
            saver.save(self.session,FLAGS.model)    
       
            actual_batch = self.session.run(get_batch)
          
            initial_cost = self.session.run(loss,feed_dict={x:actual_batch})
            cost = initial_cost
            for i in range(FLAGS.iters):
                actual_batch = self.session.run(get_batch)
		#print "*************************NEG: ",(actual_batch<0).sum()
                _, c = self.session.run([tr,loss],feed_dict={x:actual_batch})
                print "Cost at iter ",i," : ",c
                if(c<cost):
                    print "***************Best model found so far at iter ",i
                    saver.save(self.session,FLAGS.model)
                    cost = c
            actual_batch = self.session.run(get_batch)
            final_cost = self.session.run(loss,feed_dict={x:actual_batch})
        
        
        
            print "Initial cost: ",initial_cost," Final cost: ",final_cost," Best: ",c
                  
            print "Drop? ",self.use_dropout        
                  

            self.use_dropout=False        
            print "ori: ",np.mean(self.session.run(ori_c,feed_dict={x:actual_batch}),0)
            print "red: ",np.mean(self.session.run(hat_c,feed_dict={x:actual_batch}),0)
            
            #print "ba: ",actual_batch.shape
            #print "W: ", self.session.run(self.layers[0].W)

	'''
            import scipy.io as sio

            mat = sio.loadmat("./single_ex.mat")
            data = mat['a']

            data_ = np.expand_dims(data,0)
            patch = data_[:,0:7,0:7,:]
           # print patch.shape
           # print "ori: ",np.mean(self.session.run(ori_1,feed_dict={x:patch}),0)
           # print "red: ",np.mean(self.session.run(hat_1,feed_dict={x:patch}),0)
           # print "bori: ",np.mean(self.session.run(ori_1,feed_dict={x:np.expand_dims(actual_batch[0],0)}),0)
           # print "bred: ",np.mean(self.session.run(hat_1,feed_dict={x:np.expand_dims(actual_batch[0],0)}),0)
            
            print "patch:",patch
            print "batch: ",actual_batch[0]
            #print "NEG: ",(patch<0).sum()
	
	'''    

import tensorflow as tf

class layer(object):
    

    def __init__(self,units,activation=None,mean=None,std=None):
        
      
        assert not(units is None),"You need to provide the number of units ([n_in,n_out])"
        if(mean is None):
            mean=0.0
        if(std is None):
            std = 1.0
        
        
        self.W = tf.Variable(tf.truncated_normal(units,mean=mean,stddev=std))
        self.b = tf.Variable(tf.zeros([units[1]]))
        self.n_in,self.n_out = units
        
              
        if(activation is None):
            self.activation = 'sigmoid'
        else:
            self.activation = activation
            
        
        
            
    def assign_W(self,W,T=False):
        if(not T):
            assert W.get_shape().as_list() == [self.n_in,self.n_out], "W must be tensorflow.Variable with the same size of [n _in,n_out]!"
            
            self.W = tf.Variable(W.initialized_value())
        else:
            assert W.get_shape().as_list() == [self.n_out,self.n_in], "W must be tensorflow.Variable with the same size of [n _in,n_out]!"
            
            self.W = tf.Variable(tf.transpose(W.initialized_value()))
    
    def assign_b(self,b):
        
            assert b.get_shape().as_list() == [self.n_out], "b must be tensorflow.Variable with the same size of [n_out]!" 
            
            self.b = tf.Variable(b.initialized_value())
            
      
    def assign(self,W,b,T=False):
        self.assign_W(W,T)
        self.assign_b(b)
            
        
            
    def output(self,x):
        if(self.activation == 'sigmoid'):
            return tf.nn.sigmoid(tf.matmul(x,self.W)+self.b)
        elif(self.activation == 'relu'):
            return tf.nn.relu(tf.matmul(x,self.W)+self.b)
        elif(self.activation == 'linear'):
            return tf.matmul(x,self.W)+self.b
        elif(self.activation == 'softplus'):
            return tf.nn.softplus(tf.matmul(x,self.W)+self.b)
        elif(self.activation == 'tanh'):
            return tf.tanh(tf.matmul(x,self.W)+self.b)
        else:
            print "No known activation function selected, using linear"
            return tf.matmul(x,self.W)+self.b
            
        
        

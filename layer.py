import tensorflow as tf

class layer(object):
    

    def __init__(self,units,activation=None,mean=None,std=None,eur=False,graph=None):
        
        if(graph is None):
            graph = tf.Graph()
      
        self.graph = graph   
      
        with self.graph.as_default():      
              
            assert not(units is None),"You need to provide the number of units ([n_in,n_out])"
            if(mean is None):
                mean=0.0
            if(std is None):
                std = 1.0/(float(units[0])**0.5)
        
        
            self.n_in,self.n_out = units
      
      
              
            if(activation is None):
                self.activation = 'sigmoid'
            else:
                self.activation = activation
        
            if(eur):
                if(self.activation =='sigmoid'):
                    self.W = tf.Variable(tf.random_uniform(units,minval=(-4*(6.0/(self.n_in+self.n_out))**0.5),maxval=(4*(6.0/(self.n_in+self.n_out))**0.5)))
                elif(self.activation == 'relu' or self.activation == 'relu6'):
                    self.W = tf.Variable(tf.random_uniform(units,minval=0,maxval=(6.0/(self.n_in+self.n_out))**0.5))
                elif(self.activation == 'tanh'): 
                    self.W = tf.Variable(tf.random_uniform(units,minval=(-(6.0/(self.n_in+self.n_out))**0.5),maxval=((6.0/(self.n_in+self.n_out))**0.5)))
                else:
                    self.W = tf.Variable(tf.truncated_normal(units,mean=mean,stddev=std))
            else:   
                self.W = tf.Variable(tf.truncated_normal(units,mean=mean,stddev=std))
            
            self.b = tf.Variable(tf.zeros([units[1]]))
        
        
        
            
    def assign_W(self,W,T=False):
        with self.graph.as_default(): 
            if(not T):
                assert W.get_shape().as_list() == [self.n_in,self.n_out], "W must be tensorflow.Variable with the same size of [n _in,n_out]!"
                
                self.W = tf.Variable(W.initialized_value())
            else:
                assert W.get_shape().as_list() == [self.n_out,self.n_in], "W must be tensorflow.Variable with the same size of [n _in,n_out]!"
            
                self.W = tf.Variable(tf.transpose(W.initialized_value()))
    
    def assign_b(self,b):
         with self.graph.as_default(): 
            assert b.get_shape().as_list() == [self.n_out], "b must be tensorflow.Variable with the same size of [n_out]!" 
            
            self.b = tf.Variable(b.initialized_value())
            
      
    def assign(self,W,b,T=False):
        self.assign_W(W,T)
        self.assign_b(b)
            
        
            
    def output(self,x):
        with self.graph.as_default(): 
            if(self.activation == 'sigmoid'):
           
                return tf.nn.sigmoid(tf.matmul(x,self.W+self.b))
            elif(self.activation == 'relu'):
               
                return tf.nn.relu(tf.matmul(x,self.W+self.b))
            elif(self.activation == 'relu6'):
           
                return tf.nn.relu6(tf.matmul(x,self.W+self.b))
            elif(self.activation == 'linear'):
           
                return tf.matmul(x,self.W)+self.b
            elif(self.activation == 'softplus'):
            
                return tf.nn.softplus(tf.matmul(x,self.W+self.b))
            elif(self.activation == 'tanh'):
           
                return tf.tanh(tf.matmul(x,self.W+self.b))
            else:
                print "No known activation function selected, using linear"
                return tf.matmul(x,self.W)+self.b
            
    def output_dropout(self,x,keep_prob=0.5):
       
        with self.graph.as_default():        
            if(self.activation == 'sigmoid'):
                return tf.nn.dropout(tf.nn.sigmoid(tf.matmul(x,self.W+self.b)), keep_prob)
            
            elif(self.activation == 'relu'):
                return tf.nn.dropout(tf.nn.relu(tf.matmul(x,self.W+self.b)), keep_prob)
           
            elif(self.activation == 'relu6'):
        
                return tf.nn.dropout(tf.nn.relu6(tf.matmul(x,self.W+self.b)), keep_prob)
            
           
            elif(self.activation == 'linear'):
            
                return tf.nn.dropout(tf.matmul(x,self.W)+self.b,keep_prob)
           
            elif(self.activation == 'softplus'):
           
                return tf.nn.dropout(tf.nn.softplus(tf.matmul(x,self.W+self.b)),keep_prob)
      
            elif(self.activation == 'tanh'):
         
                return tf.nn.dropout(tf.tanh(tf.matmul(x,self.W+self.b)),keep_prob)
         
            else:
                print "No known activation function selected, using linear"
            return tf.matmul(x,self.W)+self.b
 
""" An rbm implementation for TensorFlow, based closely on the one in Theano """

import tensorflow as tf
import math

def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions"""
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))

class rbm(object):
	def __init__(self, name, input_size, output_size,activation='sigmoid',euris=False,learning_rate=0.1):
		with tf.name_scope("rbm_" + name):
			if(euris):
				if(activation=='sigmoid'):
					self.weights = tf.Variable(tf.random_uniform([input_size,output_size],minval=(-4*(6.0/(input_size+output_size))**0.5),maxval=(4*(6.0/(input_size+output_size))**0.5)), name="weights")
				elif(activation=='relu' or activation=='relu6'):
					self.weights = tf.Variable(tf.random_uniform([input_size,output_size],minval=0,maxval=(6.0/(input_size+output_size))**0.5))
				elif(activation=='tanh'):
					self.weights = tf.Variable(tf.random_uniform([input_size,output_size],minval=(-(6.0/(input_size+output_size))**0.5),maxval=((6.0/(input_size+output_size))**0.5)))
				else:
					self.weights = tf.Variable(tf.truncated_normal([input_size, output_size],stddev=1.0 / math.sqrt(float(input_size))), name="weights")
			else:
				self.weights = tf.Variable(tf.truncated_normal([input_size, output_size],stddev=1.0 / math.sqrt(float(input_size))), name="weights")
			self.v_bias = tf.Variable(tf.zeros([input_size]), name="v_bias")
			self.h_bias = tf.Variable(tf.zeros([output_size]), name="h_bias")
			self.activation = activation
			self.init = False
			self.learning_rate = learning_rate
			
			
	
	def propup(self, visible):
		""" P(h|v) """
		if(self.activation == 'sigmoid'):
			return tf.nn.sigmoid(tf.matmul(visible, self.weights) + self.h_bias)
		elif(self.activation == 'relu'):
			return tf.nn.relu(tf.matmul(visible, self.weights) + self.h_bias)
		elif(self.activation == 'relu6'):
			return tf.nn.relu6(tf.matmul(visible, self.weights) + self.h_bias)
		elif(self.activation == 'linear'):
			return tf.matmul(visible, self.weights) + self.h_bias
		elif(self.activation == 'softplus'):
			return tf.nn.softplus(tf.matmul(visible, self.weights) + self.h_bias)
		elif(self.activation == 'tanh'):
			return tf.tanh(tf.matmul(visible, self.weights) + self.h_bias)
		else:
			print "No known activation function selected, using linear"
		return tf.matmul(visible, self.weights) + self.h_bias
	
	
	def propdown(self, hidden):
		""" P(v|h) """
		if(self.activation == 'sigmoid'):
			return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)
		elif(self.activation == 'relu'):
			return tf.nn.relu(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)
		elif(self.activation == 'relu6'):
			return tf.nn.relu6(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)
		elif(self.activation == 'linear'):
			return tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias
		elif(self.activation == 'softplus'):
			return tf.nn.softplus(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)
		elif(self.activation == 'tanh'):
			return tf.tanh(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)
		else:
			print "No known activation function selected, using linear"
		return tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias
		
		
	def sample_h_given_v(self, v_sample):
		""" Generate a sample from the hidden layer """
		return sample_prob(self.propup(v_sample))
	
	def sample_v_given_h(self, h_sample):
		""" Generate a sample from the visible layer """
		return sample_prob(self.propdown(h_sample))
	
	def gibbs_hvh(self, h0_sample):
		""" A gibbs step starting from the hidden layer """
		v_sample = self.sample_v_given_h(h0_sample)
		h_sample = self.sample_h_given_v(v_sample)
		return [v_sample, h_sample]
	
	def gibbs_vhv(self, v0_sample):
		""" A gibbs step starting from the visible layer """
		h_sample = self.sample_h_given_v(v0_sample)
		v_sample = self.sample_v_given_h(h_sample)
		return  [h_sample, v_sample]
	
	def cd1(self, visibles):
		" One step of contrastive divergence, with Rao-Blackwellization "
		h_start = self.propup(visibles)
		v_end = self.propdown(h_start)
		h_end = self.propup(v_end)
		w_positive_grad = tf.matmul(tf.transpose(visibles), h_start)
		w_negative_grad = tf.matmul(tf.transpose(v_end), h_end)
		
		update_w = self.weights.assign_add(self.learning_rate * (w_positive_grad - w_negative_grad))
		
		update_vb = self.v_bias.assign_add(self.learning_rate * tf.reduce_mean(visibles - v_end, 0))
		
		update_hb = self.h_bias.assign_add(self.learning_rate * tf.reduce_mean(h_start - h_end, 0))
		
		return [update_w, update_vb, update_hb]
		
	def reconstruction_error(self, dataset):
		""" The reconstruction cost for the whole dataset """
		err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1]) #si usa stop gradient per evitare che venga retropropagato
		return tf.reduce_sum(err * err)
	
	def init_rbm(self):
		ts = tf.Session()
		ts.run(tf.initialize_all_variables())
		self.init = True
		return ts
	
	def __enter__(self):
		return self
	def __exit__(self,exc_type, exc_value, traceback):
		pass
		
if(__name__=='__main__'):
	import numpy as np
	data = np.random.rand(5000,784).astype("float32")
	
	#data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]).astype("float32")
	
	test = rbm('gigi',784,196,learning_rate=0.00125)
	
	ts = tf.Session()
	
	
	ts.run(tf.initialize_all_variables())
	
	print ts.run(test.propup(data))

	for i in range(20):
		ts.run(test.cd1(data))
	
	
	print ts.run(test.propup(data))
			
	
	
	
		

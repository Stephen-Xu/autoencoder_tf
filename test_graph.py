from classifier import classifier
import numpy as np
import tensorflow as tf


'''
units = [7,7,640,64]   #################
act = ['leaky_relu6','tanh','linear']

cl = classifier(units,act)

cl.generate_classifier()
session = cl.init_network()
'''

with open("converted.mdl.meta", mode='rb') as f:
  fileContent = f.read()

print fileContent

graph_def = tf.GraphDef()

graph_def.ParseFromString(fileContent)

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  
  lo = sess.graph.get_operations()
  
  for l in lo:
    print l.name

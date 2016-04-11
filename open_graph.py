import tensorflow as tf


with open("out_graph", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()

graph_def.ParseFromString(fileContent)


tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"

graph = tf.get_default_graph()

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  
  
  layer = graph.get_tensor_by_name("import/recon/layer0/W:0")
  
  print sess.run(layer)
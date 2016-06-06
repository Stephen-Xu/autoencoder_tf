import tensorflow as tf




with open("out_graph", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()

graph_def.ParseFromString(fileContent)


tf.import_graph_def(graph_def)

print "graph loaded from disk"
graph = tf.get_default_graph()



with tf.Session() as sess:

	#vars = [op.outputs[0] for op in graph.get_operations() if op.type == "Variable"]
	#print "vars: ",vars

	#sess.run("import/recon/init_1")		
	#init = tf.initialize_all_variables()
	#print graph.all_variables()
	#saver = tf.train.Saver()

	#saver.restore(sess,"./converted.mdl")
	#sess.run(init)
	
	#sess.run(vars)
	#saver = tf.train.Saver()


	#u = graph.get_tensor_by_name("import/recon/save/Const:0")

	#print "result: ",sess.run(u)
	#print "res_op: ",sess.run(u.op)
#	
 	w = graph.get_tensor_by_name("import/recon/save/Assign:0")

	print "assign: ",sess.run(w)
	#print sess.run(u,feed_dict={model:"./converted.mld"})
	
	#import/recon/save/Const:0 = "./converted.mdl"

	#saver.restore(sess,"converted.mdl")
	lista = sess.graph.get_operations()
  
	#for l in lista:
	#	print l.name

	loaders = [l for l in lista if "save/Assign" in l.name]

	print len(loaders)

	for l in loaders:
		print sess.run(l.name+":0")

	s = graph.get_operation_by_name("import/recon/save/restore_all")
	ss = graph.get_operation_by_name("import/recon/save/save")
	print ss
	
	#print sess.run(ss)
	print "############################################"
#	print sess.run(s)
	
	#list_load = graph.get_tensor_by_name("import/recon/save/save/tensor_names:0")
	
	#print list_load
	#print sess.run(list_load)
	#lung_lista = len(sess.run(list_load))

	#for i in range(lung_lista):
	#	print list_load[i]
	
	#	sess.run(s)
	#	for i in range(len(list_load)):
	#		sess.run(list_load[i])
	#sess.run(s,feed_dict={model:"./converted.mdl"})
#	re = graph.get_operation_by_name("import/recon/init")
#	print sess.run(re)
	#layer = graph.get_tensor_by_name("import/recon/layer0/W:0")
	layer = graph.get_tensor_by_name("import/recon/W:0")
	#"import/recon/layer0/W:0")
	print sess.run(layer)
#		saver = tf.train.Saver()
	#print sess.run(graph.get_tensor_by_name("import/recon/save/Const:0"))
#	print sess.run(layer)
	print "cazzillo: "
	print sess.run(graph.get_tensor_by_name("import/recon/save/restore_slice/tensor_name:0"))


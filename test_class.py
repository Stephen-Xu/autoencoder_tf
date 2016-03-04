from classifier import classifier
import numpy as np
from scipy import misc
import sys


units = [24,72,120,120,96]   #################
act = ['tanh','tanh','linear','linear']


cl = classifier(units,act)

cl.generate_classifier(std_w=3.0,euris=True,dropout=True,keep_prob_dropout=[1.0,0.5,0.5,1.0,1.0])

session = cl.init_network()

if(sys.argv[1]=='t'):

	cl.train()

else:


	image = misc.imread('./cat.jpg').astype("float32")

	cl.stop_dropout()

	cl.load_model('./converted.mdl')

	r,o  = cl.test_model(image,session=session,model=None)

	r = np.ndarray.flatten(r)
	o = np.ndarray.flatten(o)
	print zip(r,o)
	print "dist:",np.sum((r-o)**2)
	print "red norm:",np.linalg.norm(r)
	print "ori norm:",np.linalg.norm(o)

	for i in range(len(cl.layers)):
	    print "W",i+1,":",np.linalg.norm(session.run(cl.layers[i].W))




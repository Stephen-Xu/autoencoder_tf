from classifier import classifier
import numpy as np
from scipy import misc

units = [24,48,72,120,96]   #################
act = ['relu','relu','relu','relu']


cl = classifier(units,act)

cl.generate_classifier(dropout=True,keep_prob_dropout=[1.0,0.5,0.5,0.5,1.0])

session = cl.init_network()

cl.train()

image = misc.imread('./cat.jpg').astype("float32")

cl.stop_dropout()

r,o  = cl.test_model(image,session=session,model='./converted.mdl')

r = np.ndarray.flatten(r)
o = np.ndarray.flatten(o)
print zip(r,o)
print "dist:",np.mean(abs(r-o))
print np.linalg.norm(r)
print np.linalg.norm(o)

for i in range(len(cl.layers)):
    print "W",i+1,":",np.linalg.norm(session.run(cl.layers[i].W))

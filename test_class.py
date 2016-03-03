from classifier import classifier
import numpy as np
from scipy import misc

units = [24,48,72,120,96]   #################
act = ['tanh','tanh','tanh','tanh']


cl = classifier(units,act)

cl.generate_classifier(dropout=True,keep_prob_dropout=[1.0,0.5,0.5,0.5,1.0])

session = cl.init_network()

cl.train()

image = misc.imread('./cat.jpg').astype("float32")

r,o  = cl.test_model(image,session=session,model='./converted.mdl')

print zip(r,o)
print np.mean(abs(r-o))

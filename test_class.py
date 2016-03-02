from classifier import classifier
import numpy as np
from scipy import misc

units = [24,48,96]   #################
act = ['tanh','tanh']


cl = classifier(units,act)

cl.generate_classifier()

session = cl.init_network()

#cl.train()

image = misc.imread('./cat.jpg').astype("float32")

r,o  = cl.test_model(image,session=session,model='./converted.mdl')

print zip(r,o)
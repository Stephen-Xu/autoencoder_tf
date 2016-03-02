from classifier import classifier
import numpy as np

units = [24,48,96]   #################
act = ['tanh','tanh']


cl = classifier(units,act)

cl.generate_classifier()

session = cl.init_network()

cl.train()


from classifier import classifier
import numpy as np

units = [109*109*24,109*109*48,109*109*96]   #################
act = ['tanh','tanh']


cl = classifier(units,act)

cl.generate_classifier()

session = cl.init_network()

cl.train()


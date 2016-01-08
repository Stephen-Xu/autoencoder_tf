import tensorflow as tf
import numpy as np
from autoencoder import autoencoder
from optparse import OptionParser
from tools import get_data_from_minst

parser = OptionParser()
parser.add_option("-u", "--hidden", dest="hidden",
                  help="number of units",default=[3,2])
parser.add_option("-a", "--activation",
                  dest="activation", default='sigmoid',
                  help="activation function")
parser.add_option("-b","--batch",dest="batch",default=False,
                  help="Using batch learning")
parser.add_option("-n","--number_batch",dest="n_batch",default=5,
                  help="Number of batches for learning")
parser.add_option("-g","--gradient",dest="gradient",default='gradient',
                  help="Optimization metodh")
parser.add_option("-l","--learning_rate",dest="learn_rate",default=0.01,
                  help="Learning rate")
parser.add_option("-i","--iters",dest="iters",default=1000,
                  help="Number of iterations")
parser.add_option("-m","--model_name",dest="model_name",default="./model.ckpt",
                  help="Filename for model file")
parser.add_option("-c","--class_label",dest="class_label",default=0,help="Class label to use")

(options, args) = parser.parse_args()



data = np.loadtxt('swiss.dat')

data = data/np.max(data)
data = data+np.min(data)
data = data.astype("float32")


units = [3,30,15,10,7,4,2]
actions = ['sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid']
auto = autoencoder(units,actions)

auto.generate_encoder(euris=True)
auto.generate_decoder()

auto.pre_train(data)

if(not options.batch):
    bat = None
else:
    from tools.data_manipulation.batch import rand_batch
    bat = rand_batch(data,int(options.n_batch))


auto.train(data,n_iters=int(options.iters),batch=bat,display=True,noise=False,gradient=options.gradient,learning_rate=float(options.learn_rate))



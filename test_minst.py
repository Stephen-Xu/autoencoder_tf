import tensorflow as tf
import numpy as np
from autoencoder import autoencoder
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-u", "--hidden", dest="hidden",
                  help="number of hidden units",default=10)
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
parser.add_option("-c","--class_label",dest="class_label",default="../datasets/mnist_labels/class0",help="Class label to use")


(options, args) = parser.parse_args()


units = [784,int(options.hidden)]
action = [options.activation]

l_rate =  options.learn_rate

grad = options.gradient

class_label = options.class_label

f = open("../datasets/train-images.idx3-ubyte","r")
arr = np.fromfile(f, '>u1', 60000 * 28 * 28).reshape((60000, 784))
max_value = 0xFF

lab = np.loadtxt(class_label)

arr = arr.astype(float)
arr -= max_value / 2.0
arr /= max_value

data = np.asarray([arr for (arr,lab) in zip(arr,lab) if(lab==1)])



print options

auto = autoencoder(units,action)

auto.generate_encoder()
auto.generate_decoder()


if(not options.batch):
    bat = None
else:
    from tools.data_manipulation.batch import seq_batch
    bat = seq_batch(data,int(options.n_batch))


auto.train(data,n_iters=int(options.iters),model_name=options.model_name,batch=bat,display=False,noise=False,gradient=options.gradient,learning_rate=float(options.learn_rate))

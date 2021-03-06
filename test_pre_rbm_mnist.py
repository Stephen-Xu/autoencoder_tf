import tensorflow as tf
import numpy as np
from autoencoder import autoencoder
from optparse import OptionParser
from tools import get_data_from_minst

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
parser.add_option("-c","--class_label",dest="class_label",default=0,help="Class label to use")


(options, args) = parser.parse_args()


units = [784,196,int(options.hidden)]
action = [options.activation for i in range(len(units)-1)]

l_rate =  options.learn_rate

grad = options.gradient


arr,lab = get_data_from_minst.get_data_from_minst()

data = np.asarray([arr for (arr,lab) in zip(arr,lab) if(lab==int(options.class_label))]).astype("float32")



print options

auto = autoencoder(units,action)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=False)

auto.init_network()

pre_train = [auto.session.run(l.W) for l in auto.layers]


auto.pre_train_rbm(data,n_iters=10,learning_rate=0.0000008,adapt_learn=True)
'''
if(not options.batch):
    bat = None
else:
    from tools.data_manipulation.batch import rand_batch
    bat = rand_batch(data,int(options.n_batch))


auto.train(data,n_iters=int(options.iters),model_name=options.model_name,batch=bat,display=False,noise=False,gradient=options.gradient,learning_rate=float(options.learn_rate))
'''
post_train =  [auto.session.run(l.W) for l in auto.layers]


print [np.mean(a - b) for a, b in zip(pre_train, post_train)]

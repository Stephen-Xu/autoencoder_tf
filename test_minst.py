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
parser.add_option("-s","--symm",dest="symm",default=False,help="Symmetric autoencoder")
parser.add_option("-r","--reg_w",dest="reg_w",default=False,help="Cost function with regularized weights")
parser.add_option("-w","--w_file",dest="w_file",default="./weight.txt",help="File for storing norm weights")
parser.add_option("-c","--class_label",dest="class_label",default=0,help="Class label to use")
parser.add_option("-p","--pre_train_learning_rate",dest="pre_learn_rate",default=0.001,
                  help="Learning rate for RBM pre-training")
parser.add_option("-d","--reg_lambda",dest="reg_lambda",default=0.05,help="regularization weight (lambda)")

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
auto.generate_decoder(symmetric=options.symm)

auto.pre_train_rbm(data,n_iters=10,learning_rate=float(options.pre_learn_rate),adapt_learn=True)

if(not options.batch):
    bat = None
else:
    from tools.data_manipulation.batch import rand_batch
    bat = rand_batch(data,int(options.n_batch))


auto.train(data,n_iters=int(options.iters),record_weight=True,w_file=options.w_file,reg_weight=options.reg_w,reg_lambda=options.reg_lambda,model_name=options.model_name,batch=bat,display=False,noise=False,gradient=options.gradient,learning_rate=float(options.learn_rate))

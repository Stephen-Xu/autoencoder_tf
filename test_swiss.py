import tensorflow as tf
import numpy as np
from autoencoder import autoencoder
from optparse import OptionParser
from tools import get_data_from_minst

parser = OptionParser()
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
parser.add_option("-e","--pre_train",dest="pre_train",default="rbm",help="Select different pre-train or none(default RBM)")
parser.add_option("-d","--reg_lambda",dest="reg_lambda",default=0.05,help="regularization weight (lambda)")
parser.add_option("-o","--drop_out",dest="drop_out",default=False,help="using dropout")
parser.add_option("-k","--keep_prob",dest="keep_prob",default=0.5,help="probability for dropout")
(options, args) = parser.parse_args()



data = np.loadtxt('swiss.dat')

data = data/np.max(data)
data = data+np.min(data)
data = data.astype("float32")


units = [3,30,15,10,4,2]
actions = ['linear','sigmoid','sigmoid','sigmoid','sigmoid']
act2 = ['sigmoid','sigmoid','sigmoid','sigmoid','linear']
auto = autoencoder(units,actions)

auto.generate_encoder(euris=True)
auto.generate_decoder(act=act2,symmetric=options.symm)

auto.init_network()
#pre_train = [auto.session.run(l.W) for l in auto.layers]


if(options.pre_train == 'rbm'):
    auto.pre_train_rbm(data,n_iters=10,adapt_learn=True,learning_rate=float(options.pre_learn_rate))
elif(options.pre_train == 'standard'):
    auto.pre_train(data)



#post_train = [auto.session.run(l.W) for l in auto.layers]

#print [np.mean(a - b) for a, b in zip(pre_train, post_train)]
'''
out = []
for i in range(auto.dec_enc_length):
    if i==0:
        out = auto.session.run(auto.layers[i].output(data))
        print np.mean(out)
    else:
        out = auto.session.run(auto.layers[i].output(out))
        print np.mean(out)
'''
if(not options.batch):
    bat = None
else:
    from tools.data_manipulation.batch import knn_batch
    bat = knn_batch(data,int(options.n_batch))


#l=[0.001,0.0001,0.00001,0.000001,0.0000001]

#for c in range(len(l)):

b,c  = auto.train(data,use_dropout=False,keep_prob=0.8,n_iters=int(options.iters),verbose=False,record_weight=True,w_file=options.w_file,reg_weight=options.reg_w,reg_lambda=options.reg_lambda,model_name=options.model_name,batch=bat,display=True,noise=False,gradient=options.gradient,learning_rate=float(options.learn_rate))
#   auto.train(data,n_iters=20,batch=bat,display=True,verbose=True,display_w=False,gradient=options.gradient,learning_rate=l[c])
print 'init: ',b,' best: ',c

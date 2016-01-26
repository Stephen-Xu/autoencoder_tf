from autoencoder import autoencoder
from tools.image import display
from tools import get_data_from_minst
from optparse import OptionParser
import numpy as np



parser = OptionParser()
parser.add_option("-l", "--label", dest="label",
                  help="class label",default=0)
parser.add_option("-m", "--model", dest="model",
                  help="model file",default="")
parser.add_option("-i", "--index", dest="index",
                  help="test index",default=0)
parser.add_option("-u", "--units", dest="units",
                  help="hidden units",default=10)
parser.add_option("-a","--activation",dest="activation",
                  help="activation function",default="sigmoid")
parser.add_option("-s","--save",dest="save",default=False,help="Save on file first 50 images")

(options, args) = parser.parse_args()


arr,lab = get_data_from_minst.get_data_from_minst()


data = np.asarray([arr for (arr,lab) in zip(arr,lab) if(lab==int(options.label))])

hidden = int(options.units)
units = [784,hidden]
action = [options.activation for i in range(len(units)-1)]


auto = autoencoder(units,action)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=True)





session = auto.init_network()



auto.load_model(options.model,session=session)
display.display(data[int(options.index)],28,28)
    
res = auto.get_output(np.asarray([data[int(options.index)]]),session=session)

spar = 0
dist_hid = np.zeros(hidden) 

for i in range(500):
    h = auto.get_hidden(np.asarray([data[i]]),session=session)
    dist_hid = h+dist_hid
    spar = spar + (np.sqrt(hidden)-(np.sum(np.abs(h))/np.sqrt(np.sum(h**2))))/(np.sqrt(hidden)-1)

print "mean sparseness of hidden: ", spar/500.0

display.plot_hist(np.transpose(dist_hid/500.0),bins=20,normed=False)

    


display.display(res,28,28)


if(options.save):
    for i in range(50):
        display.save(data[i],28,28,folder='./imgs',name='in',index=i)
    
        res = auto.get_output(np.asarray([data[i]]),session=session)

        display.save(res,28,28,folder='./imgs',name='out',index=i)



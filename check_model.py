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
parser.add_option("-a","--activation",dest="act",
                  help="activation function",default="sigmoid")

(options, args) = parser.parse_args()


arr,lab = get_data_from_minst.get_data_from_minst()


data = np.asarray([arr for (arr,lab) in zip(arr,lab) if(lab==int(options.label))])



units = [784,int(options.units)]
act = [options.act]



auto = autoencoder(units,act)

auto.generate_encoder(euris=False)
auto.generate_decoder()

print auto.units

session = auto.init_network()



auto.load_model(options.model,session=session)




display.display(data[int(options.index)],28,28)

res = auto.get_output(np.asarray([data[int(options.index)]]),session=session)

display.display(res,28,28)

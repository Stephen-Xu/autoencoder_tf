from autoencoder import autoencoder
from tools.image import display
from tools import get_data_from_minst
import numpy as np
import sys


arr,lab = get_data_from_minst.get_data_from_minst()


data = np.asarray([arr for (arr,lab) in zip(arr,lab) if(lab==sys.argv[4])])

units = [784,int(sys.argv[3])]
act = [sys.argv[5]]



auto = autoencoder(units,act)

auto.generate_encoder()
auto.generate_decoder()


session = auto.init_network()

auto.load_model(sys.argv[1],session=session)


out = auto.get_output(data,session=session)

print data.shape
print len(data)

ind = int(sys.argv[2])

for i in range(10):
    display.display(data[i],28,28)

#display.display(out[ind],28,28)
#display.display(data[ind],28,28)


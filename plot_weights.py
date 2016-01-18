import numpy as np
import sys
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="file",
                  help="File where the weights are stored",default="./weights.txt")
parser.add_option("-l", "--layer",
                  dest="layer", default=0,
                  help="Selected layer")


(options, args) = parser.parse_args()


r_w = np.loadtxt(options.file)

l = int(options.layer)

plt.plot([w[l] for w in r_w])

plt.show()
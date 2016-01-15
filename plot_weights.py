import numpy as np
import sys
import matplotlib.pyplot as plt
        


r_w = np.loadtxt(sys.argv[1])

l = int(sys.argv[2])

plt.plot([w[l] for w in r_w])

plt.show()
import matplotlib.pyplot as plt
import numpy as np

def display(vect,w,h):
    
    img = np.reshape(vect,(w,h))


    imgplot = plt.imshow(img, interpolation="bicubic")
    
    plt.show()
    
    
    
    
    
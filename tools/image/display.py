import matplotlib.pyplot as plt
import numpy as np

def display(vect,w,h):
    
    img = np.reshape(vect,(w,h))


    imgplot = plt.imshow(img, interpolation="bicubic")
    
    plt.show()
    
    
    
def save(vect,w,h,index=0,name='foo',folder='./'):
    
    
    img = np.reshape(vect,(w,h))
    plt.imshow(img, interpolation="bicubic")
    
    plt.savefig(folder+'/'+str(index)+'_'+name+'.png')
    
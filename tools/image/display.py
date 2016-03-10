import matplotlib.pyplot as plt
import numpy as np

def display(vect,w,h,c=0):
    
    
    vect = vect+abs(np.min(vect))
    
    vect = vect/np.max(vect)*255
    
    vect = vect.astype(np.uint8)
    
    if(c==0):
        img = np.reshape(vect,(w,h))
    else:
        img = np.reshape(vect,(w,h,c))

    imgplot = plt.imshow(img, interpolation="bicubic")
    
    plt.show()
    
    
    
def save(vect,w,h,c=0,index=0,name='foo',folder='./'):
    
    vect = vect+abs(np.min(vect))
    
    vect = vect/np.max(vect)*255
    
    vect = vect.astype(np.uint8)
    if(c==0):
        img = np.reshape(vect,(w,h))
    else:
        img = np.reshape(vect,(w,h,c))
    plt.imshow(img, interpolation="bicubic")
    
    plt.savefig(folder+'/'+str(index)+'_'+name+'.png')
    
    
def plot_hist(vect,bins=50,normed=True):
    plt.hist(vect, bins=bins, normed=normed)
    plt.show()
    
    
def plot_vect(vect):
    plt.plot(vect)
    plt.show()
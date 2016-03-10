import imageflow as iw
from scipy import misc
from os import listdir
import numpy as np
from os.path import isfile, join

def conv_images():
    mypath = '../datasets/ILSVRC2012_VAL_SET/pre_images/'
  
   
    files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    files = files[:1000]
    
    image = np.asarray([np.expand_dims(misc.imread(f),0) for f in files])
   
    print image.shape
    
    iw.convert_images(image,np.asarray(range(len(files))),"./data")
    
    
    
def make_batch():
   
    
    val_images, val_labels = iw.inputs(filename='../data.tfrecords', batch_size=50,
                                    num_epochs=1,
                                    num_threads=3, imshape=[224, 224, 3])
    return val_images
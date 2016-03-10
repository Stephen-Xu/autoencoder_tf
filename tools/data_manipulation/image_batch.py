import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np




def image_batch_queue(mypath,batch_size=50,h=224,w=224):
    
    files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    file_queue = tf.train.string_input_producer(files, shuffle=False, capacity=len(files))
    image = read_image(file_queue,h,w)
    print image
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 1 * batch_size
    im_batch = tf.train.shuffle_batch(
      [image], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue,num_threads=1)
    
    
    print "batch finished"
    return im_batch, file_queue




def read_image(f_image,h=224,w=224):
    reader = tf.WholeFileReader()
    key,value = reader.read(f_image)

    image = tf.image.decode_jpeg(value)
    
    #image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    #image.set_shape([h,w,3])


    #image = tf.expand_dims(image,[0])
    
    return image





def image_batch(mypath,batch_size=50,h=224,w=224):
    
    files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    n = len(files)
    n = 100
    
    num_batch = int(np.floor(n/batch_size))
    
    file_list = tf.constant(files[:n])
    
    file_queue = tf.train.string_input_producer(file_list, num_epochs=1,shuffle=True, capacity=n)
    
    
    image_batch = []
    for i in range(num_batch):
        temp = []
        print "batch....", float(i)/num_batch*100
        for j in range(batch_size):
            image,label = read_image(file_queue,h,w)
            temp.append(image)
            
        image_batch.extend(temp)
            
    
    print "finish batch returning image"
    return image_batch
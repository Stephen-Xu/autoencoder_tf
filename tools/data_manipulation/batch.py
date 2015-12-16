import numpy as np
from tools.geometric.knn import knn


def seq_batch(data,n):
    assert n<=len(data),'The number of batches must be smaller than the data cardinality'
    
    batch_size = np.floor(len(data)/n)
    last_batch = len(data)%n
    batch = []
    for i in range(n):
        batch.append(data[i*batch_size:(i+1)*batch_size-1,:])
        #print data[i*batch_size:(i+1)*batch_size-1,:],"....",'\n'
    if(last_batch):
        #batch[i+1] = data[n*batch_size:(n*batch_size+last_batch-1),:]
        batch.append(data[n*batch_size:(n*batch_size+last_batch-1),:])
    return batch

def rand_batch(data,n):
    assert n<=len(data),'The number of batches must be smaller than the data cardinality'
    p = range(len(data))
    index = np.random.permutation(p)
    
    batch_size = np.floor(len(data)/n)
    last_batch = len(data)%n
    batch = []
    for i in range(n):
        batch.append(data[index[i*batch_size:(i+1)*batch_size-1],:])
        #print data[i*batch_size:(i+1)*batch_size-1,:],"....",'\n'
    if(last_batch):
        #batch[i+1] = data[n*batch_size:(n*batch_size+last_batch-1),:]
        batch.append(data[index[n*batch_size:(n*batch_size+last_batch-1)],:])
    return batch

def knn_batch(data,k):
    batch=[]
    
    
    _,_,ind = knn(data,k)
    
    for i in range(data.shape[0]):
        batch.append(data[ind[i],:])
    
    return batch



    

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter


def knn(data,k):
    assert k<=len(data)-1, 'The number of neighbors must be smaller than the data cardinality (minus one)'
    k=k+1
    n,dimension = data.shape
    ind = []
    dist = []
    

    if(dimension<10):
        rbp = RandomBinaryProjections('rbp', dimension)
    else:
        rbp = RandomBinaryProjections('rbp',10)
        
    engine = Engine(dimension, lshashes=[rbp], vector_filters=[NearestFilter(k)])

    for i in range(n):
        engine.store_vector(data[i], i)
    
    
    for i in range(n):
     
        N = engine.neighbours(data[i])
        ind.append([x[1] for x in N][1:])
        dist.append([x[2] for x in N][1:])
        
  
    return N,dist,ind


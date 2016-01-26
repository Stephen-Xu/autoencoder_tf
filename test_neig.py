import tensorflow as tf
import numpy as np
from tools.geometric import knn


k = 7
tau = 2

dim = 3

n = 20

data = np.random.randn(n,dim)

#_,dist,ind_www = knn.knn(data,k)

dist_knn = np.random.randn(n,k)



#print [len(i) for i in ind]

x = tf.placeholder("float",[n,dim])


neigh_index = tf.placeholder("int32",[n,k])
rep_index =  tf.placeholder("int32",[n,k])
dist = tf.placeholder("float32",[n,k])

s_weight = tf.exp(tf.pow(-dist,2)/tau)

s_w_rep = tf.reshape(tf.tile(s_weight,[1,dim]),[n,k,dim])

x_neig = tf.gather(x,neigh_index)
x_rep = tf.gather(x,rep_index)


cost = tf.reduce_mean(tf.pow(x_neig-x_rep,2)*s_w_rep)

#cost = tf.reduce_mean(x_rep*s_w_rep)

session = tf.Session()

init = tf.initialize_all_variables()

session.run(init)

ind = []
for i in range(n):
    p = np.random.permutation(n)
    ind.append(list(p[:k]))

#print ind
res = session.run(x_neig,feed_dict={x:data,neigh_index:ind})

s_ind = []
for i in range(n):
    s_ind.append(list(np.tile(i,k)))
    
rep = session.run(x_rep,feed_dict={x:data,rep_index:s_ind})


test = session.run(s_w_rep,feed_dict={dist:dist_knn,rep_index:s_ind})

final = session.run(cost,feed_dict={x:data,neigh_index:ind,rep_index:s_ind,dist:dist_knn})

print '-----'
#print res
print res.shape
print '-----'
#print rep
print rep.shape
print '-----'
#print final'''
print test.shape
#print test


print "cost: ",final
'''


n_test = np.random.rand(n,k,dim)

#print dist, ind

#for i in range(data.shape[0]):
 #  print len(dist[i])

x_k = tf.expand_dims(x,0)
neigh = tf.placeholder("float",[None,k,dim])
distance = tf.placeholder("float",[None,k])


####basta dare l'index del neighborhood in una variable


#x_neig = tf.reshape(tf.gather(x,[0,0,0,0]),[4,n,dim])

ind_list = tf.placeholder("int32",[None,4])

x_neig = tf.reshape(tf.gather(x,ind_list),[4,1,dim])



#c = tf.reduce_sum(ind_list)
#c = tf.tile(x,2)
inddd = np.asarray([[1,2,3,4]]).astype(int)


session = tf.Session()

init = tf.initialize_all_variables()

session.run(init)

res = session.run(x_neig,feed_dict={x:data,ind_list:inddd})





#ind = session.run(c,feed_dict={x:data}) #ind_list:inddd})

print res.shape
print res
#print ind
#print data
'''
'''
rec_weigh = tf.transpose(tf.reshape(tf.tile(tf.exp(-tf.pow(distance,2)/tau),[k]),[k,k]))  #CHECK VALORI SONO MOOOOOLTO PROSSIMI AD UNO
 

diff = tf.sqrt(tf.pow(tf.tile(x,[k,1])-neigh,2))   #distanza euclidea!!!!

cost = tf.reduce_sum(tf.matmul(rec_weigh,diff))

    
sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)


c=0

for i in range(100):
    
    
 
#   if(len(dist[i])==k and c==0):
 #       co = sess.run(cost,feed_dict={distance:dist[i],x:data[i:i+1],neigh:data[ind[i]]})
   #     print "ratio: ",np.std(data[ind[i]])/co
        #print data[i]
        #print data[ind[i]]
        #print sess.run(fin,feed_dict={distance:dist[i],#:data[i:i+1],neigh:data[ind[i]]})'''
     

'''
'''

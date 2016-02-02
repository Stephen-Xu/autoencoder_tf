from sklearn import neighbors
import numpy as np


class knn_classifier(object):
    
    def __init__(self,data,label,k=8,weights='uniform'):

        self.x = data
        self.y = label
        self.k = k
        self.clf = neighbors.KNeighborsClassifier(k,weights=weights)
        
    def learn(self):
        self.clf.fit(self.x,self.y)
        pass
        
    def predict(self,val):
        return self.clf.predict(val)
         
    def accuracy(self,val):
        z = self.predict(val)
        
        return np.sum(np.equal(self.y,z))/(float(len(self.y)))
        
         
        
        
    


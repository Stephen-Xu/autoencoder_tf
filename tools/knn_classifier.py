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
         
    def accuracy(self,val,lab=None,index=None):
        z = self.predict(val)
        if lab is None:
		lab = self.y

	if index is None:
		label = lab
	else:
		label = [lab[i] for i in index]
        return np.sum(np.equal(label,z))/(float(len(label)))
        
         
        
        
    


from tools import knn_classifier

from sklearn import neighbors, datasets

n_neighbors = 1

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

k = knn_classifier.knn_classifier(data=X,label=y,k=15,weights='distance')

k.learn()

print zip(k.predict(X),y)


print k.accuracy(X)
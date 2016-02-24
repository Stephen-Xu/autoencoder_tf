from tools import knn_classifier
from tools import get_data_from_minst
from sklearn import neighbors, datasets

n_neighbors = 3

# import some data to play with
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
#y = iris.target

data, label = get_data_from_minst.get_data_from_minst()
test, labtest = get_data_from_minst.get_test_from_mnist()

k = knn_classifier.knn_classifier(data=data,label=label,k=3)

k.learn()

print zip(k.predict(test[:100]),labtest[:100])


print k.accuracy(test[:5000],labtest[:5000])

from autoencoder import autoencoder
from tools.image import display
from tools import get_data_from_minst
from tools import knn_classifier
from optparse import OptionParser
import numpy as np



parser = OptionParser()
parser.add_option("-m", "--model", dest="model",
                  help="model file",default="")
parser.add_option("-i", "--index", dest="index",
                  help="test index",default=0)
parser.add_option("-u", "--units", dest="units",
                  help="hidden units",default=10)
parser.add_option("-a","--activation",dest="activation",
                  help="activation function",default="sigmoid")
parser.add_option("-s","--save",dest="save",default=0,help="Save on file first 50 images")
parser.add_option("-y","--classifier",dest="classifier",default=0,help="using nearest neighbor classifier")
(options, args) = parser.parse_args()

data = np.loadtxt("../datasets/multi_pie.dat")
label = np.loadtxt("../datasets/multi_pie_label.dat")

data = data+abs(np.min(data))
data = data/np.max(data)
data = data.astype("float32")

hidden = int(options.units)
units = [5600,1100,200,hidden]

act = ['sigmoid','sigmoid','sigmoid']


auto = autoencoder(units,act)

auto.generate_encoder(euris=True)
auto.generate_decoder(symmetric=True)





session = auto.init_network()



auto.load_model(options.model,session=session)
#display.display(data[int(options.index)],70,80)
'''    
#res = auto.get_output(np.asarray([data[int(options.index)]]),session=session)
spar = 0
dist_hid = np.zeros(hidden) 

for i in range(500):
    h = auto.get_hidden(np.asarray([data[i]]),session=session)
    dist_hid = h+dist_hid
    spar = spar + (np.sqrt(hidden)-(np.sum(np.abs(h))/np.sqrt(np.sum(h**2))))/(np.sqrt(hidden)-1)

print "mean sparseness of hidden: ", spar/500.0
print "m: ",np.asarray(dist_hid)
#display.plot_hist(np.transpose(dist_hid/500.0),bins=20,normed=False)
#display.plot_vect(np.asarray(dist_hid)/500.0)
    


#display.display(res,70,80)
'''
'''
b=np.array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,
        0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.])


v =np.array([ 0.45058036944,  0.50768027818,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.35498263446,  0.42445668094,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.48701442382,  0.,  0.,  0.,  0.,  0.,  0.69958887912,  0.,  0.,  0.,  0.,  0.,  0.,
        0.34858549299999997,  0.39739094756,  0.,  0.,  0.,  0.,  0.,  0.])
z =np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

w = auto.session.run(auto.layers[1].W)

hid = [0,1,27,28,39,45,52,53]

for j in range(8):
    z[hid[j]] = v[hid[j]]
    c = np.matmul(z,w)
    z = np.zeros(60)
    display.display(c,28,28)
m = np.mean(data[:500],axis=0)

c = np.matmul((b+v),w)'''
#display.display(c,28,28)
#display.display(np.ndarray.flatten(np.array(m)),28,28)

if(int(options.save)):
    for i in range(50):
        display.save(data[i],70,80,folder='./imgs_pie',name='in',index=i)
    
        res = auto.get_output(np.asarray([data[i]]),session=session)

        display.save(res,70,80,folder='./imgs_pie',name='out',index=i)

if(int(options.classifier)):
    train = np.loadtxt("../datasets/multi_pie_train.dat")
    l_train = np.loadtxt("../datasets/multi_pie_l_train.dat")
    test = np.loadtxt("../datasets/multi_pie_test.dat")
    l_test = np.loadtxt("../datasets/multi_pie_l_test.dat")

    train = train+abs(np.min(train))
    train = train/np.max(train)
    train = train.astype("float32")

    test = test+abs(np.min(test))
    test = test/np.max(test)
    test = test.astype("float32")
    
    h = auto.get_hidden(np.asarray(train),session=session)
    k = knn_classifier.knn_classifier(data=h,label=l_train,k=3)
    #k =  knn_classifier.knn_classifier(data=train,label=l_train,k=3)
    k.learn()

    

    t = auto.get_hidden(np.asarray(test),session=session)
    
    print k.accuracy(t,l_test)
    print zip(k.predict(t),l_test)
    #print k.accuracy(test,l_test)


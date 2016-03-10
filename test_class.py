from classifier import classifier
import numpy as np
from scipy import misc
import sys
from tools.image import display


units = [24,72,120,120,96]   #################
act = ['tanh','tanh','linear','linear']


with tf.Graph().as_default():

    cl = classifier(units,act)

    cl.generate_classifier(std_w=3.0,euris=True,dropout=True,keep_prob_dropout=[1.0,0.5,0.5,1.0,1.0])

    session = cl.init_network()

    if(sys.argv[1]=='t'):

    
    	cl.train()

    else:
    
        image = misc.imread('./cat.jpg').astype("float32")
    
        print np.mean(session.run(cl.layers[0].W))
        
        cl.stop_dropout()
    
        image = np.random.uniform(1,255,[7,7,3])
    
        cl.load_model('./converted.mdl',session=session)
    
        print np.mean(session.run(cl.layers[0].W))
    
        r,o,p  = cl.test_model(image,session=session)
    
        r = np.ndarray.flatten(r)
        o = np.ndarray.flatten(o)
    #print zip(r,o)
        print "dist:",0.5*np.sum((r-o)**2)
        print "red norm:",np.linalg.norm(r)
        print "ori norm:",np.linalg.norm(o)
    
        for i in range(len(cl.layers)):
            print "W",i+1,":",np.linalg.norm(session.run(cl.layers[i].W))

   # p = np.squeeze(p)
    #display.display(p,p.shape[0],p.shape[1],c=3)
    #print p.shape

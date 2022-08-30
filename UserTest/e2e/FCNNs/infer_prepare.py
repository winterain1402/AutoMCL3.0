import numpy as np
def prepare():
    for i in [1,16,32,64,128,256,512]:
        tt_array = np.array([0,0]).reshape((-1,2))
        np.savetxt("batch="+str(i)+".txt",tt_array)

prepare()
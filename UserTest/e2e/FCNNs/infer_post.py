import numpy as np
for i in [1,16,32,64,128,256,512]:
    my_arr = np.loadtxt("batch=" + str(i) + ".txt")
    #print(my_arr.shape)
    print("batch=",i,"[warmup,avg]",my_arr[1:-1,:].mean(axis=0))

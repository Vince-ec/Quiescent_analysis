import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time


test=open('pickled_mstacks/com_mar26_rfv_spec.pkl','rb')

# diff=[]
# for i in range(len(age)*len(tau)*len(metal)):
mf1 = cPickle.load(test)
print len(mf1)
print mf1
plt.plot(mf1)
plt.show()

test.close()
burst.close()
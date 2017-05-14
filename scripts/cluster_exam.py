import numpy as np
from spec_id import Cluster_fit

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,14.1,.1)
tau=[0]

cluster=[6528,6553,5927,6304,6388,6441]

for i in range(len(cluster)):
    Cluster_fit('clusters/ngc%s_griz_cer.npy' % cluster[i],metal,age,'ngc%s_flater' % cluster[i],flat_err= True)
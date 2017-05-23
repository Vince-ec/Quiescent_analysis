import numpy as np
from spec_id import Cluster_fit

metal=np.arange(0.002,0.031,0.001)
# age=np.arange(.5,14.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
# tau = [0]

age=np.arange(5,13.1,.1)
rshift=[1.1,1.2,1.35]
cluster=[6528, 6553,5927,6304,6388,6441]

# for i in range(len(cluster)):
#     for ii in range(len(rshift)):
#           Cluster_fit('../clusters/ngc%s_griz_err_%s.npy' % (cluster[i],rshift[ii]), metal, age, tau, rshift[ii], 'ngc%s_err_al_%s' % (cluster[i],rshift[ii]))


age=[np.arange(10,14.1,.1),np.arange(8,14.1,.1),np.arange(9.9,14.1,.1),np.arange(11.3,14.1,.1),np.arange(11.6,14.1,.1),np.arange(12.5,14.1,.1)]

for i in range(len(cluster)):
    for ii in range(len(rshift)):
          Cluster_fit('../clusters/ngc%s_griz_err_%s.npy' % (cluster[i],rshift[ii]), metal, age[i], tau, rshift[ii], 'ngc%s_err_va_%s' % (cluster[i],rshift[ii]))
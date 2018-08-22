import numpy as np
from C_MC_fit import MC_fit

metal=np.round(np.arange(0.002,0.031,0.001),3)
age=np.round(np.arange(.5,6.1,.1),1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
mtest=np.round(np.arange(0.002,0.031,0.008),3)
atest=np.arange(.5,6.1,.5)
tau_test=[0,8.0, 8.3, 8.48, 8.6]
ztest = np.array([1.219,1.220])

MC_fit('s40597', mtest, age, tau, ztest, np.arange(0, 1.1, 0.1), 0.018, 2.0, 0, 
       1.22, 0.0, 10, 'all_test', 1.221, 'mc_test', repeats=10)
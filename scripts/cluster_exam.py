import numpy as np
from spec_id import Cluster_fit

metal=np.arange(0.002,0.031,0.001)
# age=np.arange(.5,14.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
# tau = [0]

cluster=[6528,6553,5927,6304,6388,6441]

# age = [np.arange(7,13.1,.1),np.arange(8,13.1,.1),np.arange(7,13.1,.1),
#        np.arange(8,14.1,.1),np.arange(8,14.1,.1),np.arange(8,14.1,.1)]

age=np.arange(5,13.1,.1)
Cluster_fit('../clusters/ngc%s_griz_err_1.2.npy' % cluster[0], metal, age, tau, 1.2, 'ngc%s_err_al_1.2' % cluster[0])
Cluster_fit('../clusters/ngc%s_griz_err_1.35.npy' % cluster[0], metal, age, tau, 1.35, 'ngc%s_err_al_1.35' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.1.npy' % cluster[0], metal, age, [0], 1.1, 'ngc%s_err_1.1' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.2.npy' % cluster[0], metal, age, [0], 1.2, 'ngc%s_err_1.2' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.35.npy' % cluster[0], metal, age, [0], 1.35, 'ngc%s_err_1.35' % cluster[0])

age=np.arange(.5,14.1,.1)
Cluster_fit('../clusters/ngc%s_griz_err_1.2.npy' % cluster[0], metal, age, tau, 1.2, 'ngc%s_err_al_fa_1.2' % cluster[0])
Cluster_fit('../clusters/ngc%s_griz_err_1.35.npy' % cluster[0], metal, age, tau, 1.35, 'ngc%s_err_al_fa_1.35' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.1.npy' % cluster[0], metal, age, [0], 1.1, 'ngc%s_err_1.1_fa' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.2.npy' % cluster[0], metal, age, [0], 1.2, 'ngc%s_err_1.2_fa' % cluster[0])
# Cluster_fit('../clusters/ngc%s_griz_err_1.35.npy' % cluster[0], metal, age, [0], 1.35, 'ngc%s_err_1.35_fa' % cluster[0])

# age=np.arange(4,13.1,.1)
# Cluster_fit('../clusters/ngc%s_griz_err.npy' % cluster[1],metal,age,tau,'ngc%s_err_al' % cluster[1])
#
# age=np.arange(9,14.1,.1)
# Cluster_fit('../clusters/ngc%s_griz_err.npy' % cluster[2],metal,age,tau,'ngc%s_err_al' % cluster[2])
#
# age=np.arange(9,14.1,.1)
# Cluster_fit('../clusters/ngc%s_griz_err.npy' % cluster[3],metal,age,tau,'ngc%s_err_al' % cluster[3])
#
# age=np.arange(9,14.1,.1)
# Cluster_fit('../clusters/ngc%s_griz_err.npy' % cluster[4],metal,age,tau,'ngc%s_err_al' % cluster[4])
#
# age=np.arange(9,14.1,.1)
# Cluster_fit('../clusters/ngc%s_griz_err.npy' % cluster[5],metal,age,tau,'ngc%s_err_al' % cluster[5])
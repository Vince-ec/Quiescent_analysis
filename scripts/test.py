import numpy as np
from spec_id import Scale_model, Cluster, Cluster_model,Gauss_dist,Divide_cont,Cluster_fit_sim_MC
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

metal=np.arange(0.01,0.02,0.001)
age=np.arange(2.0,5.0,.1)
tau=[0,8.0, 8.3, 8.48, 8.6]

Cluster_fit_sim_MC('../clusters/ngc6528_griz_err_1.1.npy', 0.015, 4.0 , 8.6, metal, age,
                   [0], 1.1, 'test', repeats=1000)

mlist,alist = np.load('../mcerr/test_mcerr.npy')
ncmlist,ncalist = np.load('../mcerr/test_nc_mcerr.npy')

sea.kdeplot(mlist,alist,n_levels=30,cmap=colmap)
# plt.scatter(mlist,alist)
plt.plot(0.015,11,'rp')
plt.axis([0,.03,0,14])
plt.show()

sea.kdeplot(ncmlist,ncalist,n_levels=30,cmap=colmap)
# plt.scatter(ncmlist,ncalist)
plt.plot(0.015,11,'rp')
plt.axis([0,.03,0,14])
plt.show()

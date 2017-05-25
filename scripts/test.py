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

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,14.1,.1)

# Cluster_fit_sim_MC('../clusters/ngc6528_griz_err_1.1.npy',0.015,11.0,0,metal,age,1.1,'sim6528_1.1',repeats=1000)

mlist,alist = np.load('../mcerr/sim6528_1.1_mcerr.npy')
ncmlist,ncalist = np.load('../mcerr/sim6528_1.1_nc_mcerr.npy')

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

import numpy as np
from spec_id import Cluster, Cluster_model,Gauss_dist,Divide_cont,Divide_cont_model,Cluster_fit_sim_MC,Galaxy_sim,\
    Galaxy_ids,Galaxy
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
#
# gal1 = Galaxy_sim(0.015,4.0,0,1.1)
# gal1.Simulate()
# gal1.Remove_continuum(use_sim=True)
# cmodel = Galaxy_sim(0.01, 2.0, 0, 1.1)
# cmodel.Remove_continuum()
#
# plt.figure(figsize=[12, 5])
# plt.subplot(211)
# plt.plot(gal1.simwv, gal1.simfl, 'b')
# # plt.plot(gal1.simwv, gal1.simer, 'b')
# plt.plot(cmodel.wv,cmodel.mfl)
# # plt.show()
# plt.subplot(212)
# # plt.figure(figsize=[12, 5])
# plt.plot(gal1.nc_simwv, gal1.nc_simfl, 'b')
# # plt.plot(gal1.nc_simwv, gal1.nc_simer, 'b')
# plt.plot(cmodel.nc_wv,cmodel.nc_fl)
# plt.show()



# metal=np.arange(0.01,0.02,0.001)
metal=np.arange(0.002,0.031,0.001)
# age=np.arange(2.0,5.0,.1)
age=np.arange(0.5,14.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6]
#
Cluster_fit_sim_MC('../clusters/ngc6528_griz_err_1.1.npy', 0.015, 4.0 , 8.0, metal, age,
                   tau, 1.1, 'test', repeats=100,use_galaxy=True)
#
mlist,alist = np.load('../mcerr/test_mcerr.npy')
ncmlist,ncalist = np.load('../mcerr/test_nc_mcerr.npy')
tmlist,talist = np.load('../mcerr/test_t_mcerr.npy')


sea.kdeplot(mlist,alist,n_levels=30,cmap=colmap)
# plt.scatter(mlist,alist)
plt.plot(0.015,4.0,'rp')
plt.axis([0,.03,0,14])
plt.show()

sea.kdeplot(ncmlist,ncalist,n_levels=30,cmap=colmap)
# plt.scatter(ncmlist,ncalist)
plt.plot(0.015,4.0,'rp')
plt.axis([0,.03,0,14])
plt.show()

sea.kdeplot(tmlist,talist,n_levels=30,cmap=colmap)
# plt.scatter(ncmlist,ncalist)
plt.plot(0.015,4.0,'rp')
plt.axis([0,.03,0,14])
plt.show()
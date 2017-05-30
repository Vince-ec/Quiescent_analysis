import numpy as np
from spec_id import Cluster, Cluster_model,Gauss_dist,Divide_cont,Divide_cont_model,Cluster_fit_sim_MC,Galaxy_sim,\
    Galaxy_ids,Galaxy,Cluster_fit_sim_MCLH,Likelihood_contours
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
age=np.arange(0.5,6.1,.1)
# tau=[0,8.0, 8.3, 8.48, 8.6]
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
M,A=np.meshgrid(metal,age)

Cluster_fit_sim_MCLH('../clusters/ngc6528_griz_err_1.1.npy', 0.015, 4.0 , 0, metal, age,
                   tau, 1.1, 'test', repeats=100, use_galaxy=True)
#
df = np.load('../mcerr/test_LH_mcerr.npy').T
nc = np.load('../mcerr/test_nc_LH_mcerr.npy').T
tf = np.load('../mcerr/test_t_LH_mcerr.npy').T

onesig, twosig = Likelihood_contours(age, metal, df)
levels = np.array([twosig, onesig])
plt.contour(M, A, df, levels, colors='k', linewidths=2)
plt.contourf(M, A, df, 40, cmap=colmap)
plt.plot(0.015,4.0,'rp')
plt.show()

onesig, twosig = Likelihood_contours(age, metal, nc)
levels = np.array([twosig, onesig])
plt.contour(M, A, nc, levels, colors='k', linewidths=2)
plt.contourf(M, A, nc, 40, cmap=colmap)
plt.plot(0.015,4.0,'rp')
plt.show()

onesig, twosig = Likelihood_contours(age, metal, tf)
levels = np.array([twosig, onesig])
plt.contour(M, A, tf, levels, colors='k', linewidths=2)
plt.contourf(M, A, tf, 40, cmap=colmap)
plt.plot(0.015,4.0,'rp')
plt.show()

# sea.kdeplot(mlist,alist,n_levels=30,cmap=colmap)
# # plt.scatter(mlist,alist)
# plt.plot(0.015,4.0,'rp')
# plt.axis([0,.03,0,14])
# plt.show()
#
# sea.kdeplot(ncmlist,ncalist,n_levels=30,cmap=colmap)
# # plt.scatter(ncmlist,ncalist)
# plt.plot(0.015,4.0,'rp')
# plt.axis([0,.03,0,14])
# plt.show()
#
# sea.kdeplot(tmlist,talist,n_levels=30,cmap=colmap)
# # plt.scatter(ncmlist,ncalist)
# plt.plot(0.015,4.0,'rp')
# plt.axis([0,.03,0,14])
# plt.show()
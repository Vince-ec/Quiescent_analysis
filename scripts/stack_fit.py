import numpy as np
from spec_id import Model_fit_stack_normwmean_rfv, Model_fit_stack_normwmean_features_rfv, \
    Model_fit_stack_normwmean_cont_rfv, Model_fit_stack_normwmean_cont,Model_fit_stack_normwmean_features,\
    Model_fit_stack_normwmean
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

ids,speclist,lmass,rshift,rad,sig,comp=np.array(Readfile('masslist_mar22.dat',is_float=False))
lmass,rshift,rad,sig,comp=np.array([lmass,rshift,rad,sig,comp]).astype(float)

gid, rfv, iracm =Readfile('galaxy_mags.dat', is_float=False)
rfv, iracm=np.array([rfv, iracm]).astype(float)

IDc=[]  # compact sample
IDd=[]  # diffuse sample

IDmL=[]  # low mass sample
IDmH=[]  # high mass sample

for i in range(len(ids)):
    if 0.11 < comp[i]:
        IDd.append(i)
    if 0.11 > comp[i]:
        IDc.append(i)
    if 10.931 > lmass[i]:
        IDmL.append(i)
    if 10.931 < lmass[i]:
        IDmH.append(i)

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
M,A=np.meshgrid(metal,age)

#####################
Model_fit_stack_normwmean(speclist[IDc],tau,metal,age,rshift[IDc],ids[IDc],np.arange(3100,5500,10),
                              'com_mar28_fit','com_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_cont(speclist[IDc],tau,metal,age,rshift[IDc],ids[IDc],np.arange(3100,5500,10),
                                   'com_cont_mar28_fit','com_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_features(speclist[IDc],tau,metal,age,rshift[IDc],ids[IDc],np.arange(3100,5500,10),
                                       'com_feat_mar28_fit','com_mar28_spec',res=10,fsps=True)
######################
Model_fit_stack_normwmean(speclist[IDd],tau,metal,age,rshift[IDd],ids[IDd],np.arange(3450,5400,10),
                              'ext_mar28_fit','ext_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_cont(speclist[IDd],tau,metal,age,rshift[IDd],ids[IDd],np.arange(3450,5400,10),
                                   'ext_cont_mar28_fit','ext_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_features(speclist[IDd],tau,metal,age,rshift[IDd],ids[IDd],np.arange(3450,5400,10),
                                       'ext_feat_mar28_fit','ext_mar28_spec',res=10,fsps=True)
######################
Model_fit_stack_normwmean(speclist[IDmH],tau,metal,age,rshift[IDmH],ids[IDmH],np.arange(3100,5400,10),
                              'gt10.93_mar28_fit','gt10.93_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_cont(speclist[IDmH],tau,metal,age,rshift[IDmH],ids[IDmH],np.arange(3100,5400,10),
                                   'gt10.93_cont_mar28_fit','gt10.93_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_features(speclist[IDmH],tau,metal,age,rshift[IDmH],ids[IDmH],np.arange(3100,5400,10),
                                       'gt10.93_feat_mar28_fit','gt10.93_mar28_spec',res=10,fsps=True)
######################
Model_fit_stack_normwmean(speclist[IDmL],tau,metal,age,rshift[IDmL],ids[IDmL],np.arange(3100,5500,10),
                              'lt10.93_mar28_fit','lt10.93_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_cont(speclist[IDmL],tau,metal,age,rshift[IDmL],ids[IDmL],np.arange(3100,5500,10),
                                   'lt10.93_cont_mar28_fit','lt10.93_mar28_spec',res=10,fsps=True)
Model_fit_stack_normwmean_features(speclist[IDmL],tau,metal,age,rshift[IDmL],ids[IDmL],np.arange(3100,5500,10),
                                       'lt10.93_feat_mar28_fit','lt10.93_mar28_spec',res=10,fsps=True)

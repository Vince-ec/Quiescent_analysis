import numpy as np
from spec_id import Stack_model, Stack_spec, Model_fit_stack,Analyze_Stack_avgage,Analyze_Stack,Likelihood_contours, Gauss_dist, Make_model_list,\
    Stack_spec_normwmean,Stack_model_normwmean, Model_fit_stack_normwmean ,Best_fit_model
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


ids,speclist,lmass,rshift,rad,sig,comp=np.array(Readfile('lowmass_testlist.dat',is_float=False))
lmass,rshift,rad,sig,comp=np.array([lmass,rshift,rad,sig,comp]).astype(float)


metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=np.arange(.5,6.1,.1)
tau=[0,8.0,8.93,9.21,9.37,9.5,9.6,9.66,9.73,9.8,9.84,9.88,9.93,9.97,10.0]
ntau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

# ntau=[8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
#      9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

M,A=np.meshgrid(metal,age)

#######################
Model_fit_stack(speclist,tau,metal,age,rshift,ids,np.arange(3450, 5300, 10),'lmass_mar8_nc_fit',
                          'lmass_mar8_nc_spec',res=10,fsps=True)

#######################
Model_fit_stack(speclist,ntau,metal,age,rshift,ids,np.arange(3450, 5300, 10),'lmass_mar8_ncnt_fit',
                          'lmass_mar8_ncnt_spec',res=10,fsps=True)

Model_fit_stack_normwmean(speclist,ntau,metal,age,rshift,ids,np.arange(3450, 5300, 10),'lmass_mar8_nt_fit',
                          'lmass_mar8_nt_spec',res=10,fsps=True)
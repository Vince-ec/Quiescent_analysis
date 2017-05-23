import numpy as np
from spec_id import Scale_model, Cluster, Cluster_model,Gauss_dist,Divide_cont,Cluster_fit
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
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
# age=np.arange(.5,14.1,.1)
age=[np.arange(10,12.3,.1),np.arange(8,13.1,.1),np.arange(9.9,12.0,.1),np.arange(11.3,13.4,.1),np.arange(11.6,13.7,.1),np.arange(12.5,14.1,.1)]
rshift=[1.1,1.2,1.35]
cluster=[6528,6553,5927,6304,6388,6441]

ngc=Cluster('../clusters/ngc%s_griz_err_%s.npy' % (cluster[0],rshift[0]),rshift[0])
ngc.Analyze_fit('../chidat/ngc%s_err_va_%s_chidata.fits' % (cluster[0],rshift[0]),metal,age[0],tau)
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
# age=np.arange(7.0,14.1,.1)
age=np.arange(.5,14.1,.1)
# tau=[0]
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

M,A=np.meshgrid(metal,age)

cluster=[6528,6553,5927,6304,6388,6441]

ngc=Cluster('../clusters/ngc%s_griz_err.npy' % cluster[0])
ngc.Analyze_fit_FC('../chidat/ngc%s_err_ap_cont_chidata.fits' % cluster[0],
                   '../chidat/ngc%s_err_ap_cont_chidata.fits' % cluster[0],metal,age,tau)
ngc.Plot_2D_likelihood()
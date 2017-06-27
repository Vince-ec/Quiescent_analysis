import numpy as np
import pandas as pd
from spec_id import Single_gal_fit_full,Analyze_LH_cont_feat
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
from time import time

import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0]

M,A=np.meshgrid(metal,age)

print len(metal)*len(age)*len(tau)

start = time()
Single_gal_fit_full(metal, age, tau, 1.022,'s39170','39170_test')
end = time()

print end - start

P = np.load('../chidat/39170_test_tZ_pos.npy')
Z,PZ = np.load('../chidat/39170_test_Z_pos.npy')
t,Pt = np.load('../chidat/39170_test_t_pos.npy')

plt.contour(M,A,P)
plt.show()

plt.plot(Z,PZ)
plt.show()

plt.plot(t,Pt)
plt.show()
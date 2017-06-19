import numpy as np
import pandas as pd
from spec_id import RT_spec
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
from time import time
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

qgDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
gsDB = qgDB[qgDB['spec'] == True]

s_spec = RT_spec('s39170')
s_spec.Sim_spec(0.024,4.7,8.0,1.023)

plt.figure(figsize=[15,5])
plt.plot(s_spec.gal_wv,s_spec.fl)
plt.errorbar(s_spec.gal_wv,s_spec.gal_fl,s_spec.gal_er,fmt = 'o',ms = 2)
plt.show()
plt.close()
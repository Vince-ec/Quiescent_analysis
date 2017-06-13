import numpy as np
from spec_id import Galaxy_set
import pandas as pd
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

qgalDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
all_g_gals = list(qgalDB[qgalDB['in_data'] == True][qgalDB['agn'] == False]['gids'])

for i in range(len(all_g_gals)):
    galaxy = all_g_gals[i]
    gal_set = Galaxy_set(galaxy)
    if len(gal_set.one_d_list) == 0:
        print galaxy
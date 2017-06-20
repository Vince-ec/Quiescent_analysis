import numpy as np
import pandas as pd
from spec_id import RT_spec, Specz_fit
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

metal=np.array([0.002,0.01,0.02,0.03])
age=np.array([1.0,2.0,3.0, 4.0, 5.0, 6.0])
z=np.arange(1.084 - 0.3 ,1.084 + 0.3,.05)

Specz_fit('s39170',metal,age,z,'39170_test')
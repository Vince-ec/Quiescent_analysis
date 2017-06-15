import galaxy_extract as ge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

qgDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
gsDB = qgDB[qgDB['spec'] == True]

ge.FLT_search(gsDB['ra'][220],gsDB['dec'][220])
import numpy as np
from spec_id import Galaxy_set
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
import os
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

qgalDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
all_g_gals = list(qgalDB[qgalDB['in_data'] == True][qgalDB['agn'] == False]['gids'])

galaxy = all_g_gals[0]

gal_set = Galaxy_set(galaxy)
gal_set.Display_spec()

"""Save quality file"""
### make data table and assign file name
qual_dat=Table([gal_set.pa_names,gal_set.quality],names=['id','good_spec'])

if os.path.isdir('../../../../vestrada'):
    fn = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s/%s_quality.txt' % (galaxy,galaxy)
else:
    fn = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s/%s_quality.txt' % (galaxy,galaxy)

### save quality file
ascii.write(qual_dat,fn,overwrite=True)

"""Stack Galaxy"""
gal_set.Get_stack_info()
IDS = [U for U in range(len(gal_set.s_wv)) if 7900 < gal_set.s_wv[U] < 11300]

plt.figure(figsize=[15,5])
plt.plot(gal_set.s_wv[IDS], gal_set.s_fl[IDS], 'r', alpha=.5)
plt.plot(gal_set.s_wv[IDS], gal_set.s_er[IDS], '--r', alpha=.5)

if gal_set.one_d_list.size > 0:

    gal_set.Mean_stack_galaxy()

    IDX = [U for U in range(len(gal_set.wv)) if 7900 < gal_set.wv[U] <11300]

    plt.plot(gal_set.wv[IDX],gal_set.fl[IDX])
    plt.plot(gal_set.wv[IDX],gal_set.er[IDX])
plt.show()

### make data table for new stack
if os.path.isdir('../../../../vestrada'):
    n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % galaxy
else:
    n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % galaxy

if gal_set.one_d_list.size > 0:
    np.save(n_dir + '/%s_stack' % (galaxy),[gal_set.wv,gal_set.fl,gal_set.er])
else:
    np.save(n_dir + '/%s_stack' % (galaxy), [gal_set.s_wv, gal_set.s_fl, gal_set.s_er])
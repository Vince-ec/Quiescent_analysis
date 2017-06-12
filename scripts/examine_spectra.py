import numpy as np
from spec_id import Galaxy_set
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

galaxy = 's39170'
# galaxy = 'n10338'

gal_set = Galaxy_set(galaxy)
# gal_set.Display_spec()
#
# """Save quality file"""
# ### make data table and assign file name
# qual_dat=Table([gal_set.pa_names,gal_set.quality],names=['id','good_spec'])
#
# if os.path.isdir('../../../../vestrada'):
#     fn = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s/%s_quality.txt' % (galaxy,galaxy)
# else:
#     fn = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s/%s_quality.txt' % (galaxy,galaxy)
#
# ### save quality file
# ascii.write(qual_dat,fn,overwrite=True)

"""Stack Galaxy"""
###test
gal_set.quality=[1,1,1,1,1]
gal_set.Mask = [[0,0],[0,0],[0,0],[9000,9500],[0,0]]
###test
gal_set.Median_stack_galaxy()

IDX = [U for U in range(len(gal_set.wv)) if 7500 < gal_set.wv[U] <11500]

plt.plot(gal_set.wv[IDX],gal_set.fl[IDX])
plt.plot(gal_set.wv[IDX],gal_set.er[IDX])
plt.show()

# ### get stack and compare to old stack
# swv,sfl,ser=Stack_gal_spec(new_speclist,wvcover[IDX][:-1],new_mask)
# swv2,sfl2,ser2=Get_flux(s[-1])
#
#
# plt.plot(swv2,sfl2,'k',alpha=.5)
# plt.plot(swv2,ser2,'k',alpha=.5)
# plt.plot(swv,sfl,'b')
# plt.plot(swv,ser,'b')
# plt.show()
#
# ### make data table for new stack
# stack_dat=Table([swv,sfl,ser],names=['wv','flam','err'])
# fn='spec_stacks_nov29/%s%s_stack.dat' % (loc,galaxy)
# ascii.write(stack_dat,fn)



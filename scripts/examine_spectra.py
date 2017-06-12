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

gal_set = Galaxy_set('s39170')
# gal_set = Galaxy_set('n10338')

gal_set.Display_spec()

### initialize quality and pointing name arrays
quality=np.repeat(True,len(ind_gal_img))
Mask=np.zeros([len(ind_gal_img),2])
p_names=[]

### examine each spectra and assign quality
for i in range(len(ind_gal_img)):
    os.system("open " + ind_gal_img[i])
    p_names.append(ind_gal_img[i][48:74])
    wv,fl,err= Get_flux(spectra[i])
    plt.plot(wv,fl)
    plt.plot(wv,err)
    plt.xlim(8000,11500)
    plt.show()
    quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
    if quality[i] != 0:
        minput = int(input('Mask region: (0 if no mask needed)'))
        if minput != 0:
            rinput= int(input('Lower bounds'))
            linput= int(input('Upper bounds'))
            Mask[i]=[rinput,linput]
#
# """Stack spectra and save quality file"""
# ### make data table and assign file name
# qual_dat=Table([p_names,quality],names=['id','good_spec'])
# fn='spec_stacks_nov29/%s%s_quality.txt' % (loc,galaxy)
#
# ### save quality file
# ascii.write(qual_dat,fn)
#
# ### select good galaxies
# new_speclist=[]
# new_mask=[]
# for i in range(len(quality)):
#     if quality[i]==True:
#         new_speclist.append(spectra[i])
#         new_mask.append(Mask[i])
#
# ### get wavelength coverage
# wvcover=[]
# wvsum=[]
# for i in range(len(new_speclist)):
#     w,f,e=Get_flux(new_speclist[i])
#     wvsum.append(sum(w))
#     wvcover.append(w)
#
# IDX = np.argwhere(wvsum==np.max(wvsum))[0]
#
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



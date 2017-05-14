from spec_id import Simspectest
from astropy.io import ascii
from astropy.table import Table
import numpy as np
from vtl.Readfile import Readfile
from time import time

filepath = '/Users/vestrada/Desktop/axesim_practice/OUTSIM/'

metal = [22, 32, 42, 52, 62, 72]
A = [ .01, .014, .02, .03, .04, .06, .1, .14, .2, .3, .4, .6, 1, 1.4, 2, 3, 4, 6, 10, 10.4, 12, 13]
z = np.linspace(1, 2, 101)

# for i in range(len(metal)):
#     for ii in range(len(A)):
#         for iii in range(len(z)):
#             wv,fl=Simspec(filepath + 'gal_%s_%s_%s_slitless_2.SPC.fits' % (metal[i],A[ii],z[iii]))
#             dat=Table([wv,fl],names=['wavelenth','flux'])
#             ascii.write(dat,'../../../Models_for_fitting/gal_%s_%s_%s.dat' % (metal[i],A[ii],z[iii]))

wv, fl,err = Simspectest(filepath + 'gal_62_2_1.5_noise_slitless_2.SPC.fits')
dat = Table([wv, fl,err], names=['wavelenth', 'flux','error'])
ascii.write(dat, '../../../Models_for_fitting/gal_62_2_1.5_noise.dat')
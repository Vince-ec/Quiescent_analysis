from spec_id import Simspec
from astropy.io import ascii
from astropy.table import Table
from vtl.Readfile import Readfile
from time import time

start1=time()
filepath = '/Users/vestrada/Desktop/axesim_practice/OUTSIM/gal_32_0.4_1.89_slitless_2.SPC.fits'
wv,fl=Simspec(filepath)
end1=time()
# dat=Table([wv,fl],names=['wavelenth','flux'])
# ascii.write(dat,'../../../Models_for_fitting/gal_32_0.4_1.89.dat')
start2=time()
filepath2='../../../Models_for_fitting/gal_32_0.4_1.89.dat'
wv2,fl2=Readfile(filepath2,1)
end2=time()

print end1-start1
print end2-start2
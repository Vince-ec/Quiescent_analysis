import numpy as np
from spec_id import Single_gal_fit_full

from vtl.Readfile import Readfile
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

ids, speclist, lmass, rshift, rad, sig, comp = np.array(Readfile('masslist_mar22.dat', is_float=False))
lmass, rshift, rad, sig, comp = np.array([lmass, rshift, rad, sig, comp]).astype(float)


metal = np.arange(0.002, 0.031, 0.001)
age = np.arange(.5, 6.1, .1)
tau = [0, 8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
       9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
M, A = np.meshgrid(metal, age)

# glist=['n21156','s39170','n34694']
# rlist=[1.252,1.023,1.146]
# slist=['spec_stacks_jan24/n21156_stack.npy','spec_stacks_jan24/s39170_stack.npy','spec_stacks_jan24/n34694_stack.npy']

glist=['s45972']
rlist=[1.04]
slist=['spec_stacks_jan24/s45972_stack.npy']

for i in range(len(ids)):
    print ids[i]
    Single_gal_fit_full(speclist[i],tau,metal,age,rshift[i],ids[i],'%s_apr6_galfit' % ids[i])
print '!!Done!!'
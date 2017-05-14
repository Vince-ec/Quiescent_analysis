import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table

infile=open('../../../Desktop/catalogs_for_CLEAR/goodss_3dhst.v4.1.cats/Eazy/goodss_3dhst.v4.1.pz')
size=np.fromfile(infile,dtype='intc',count=2)
allpz=np.fromfile(infile,dtype='float64',count=size[0]*size[1])
pzgrid=allpz.reshape([size[1],size[0]])
ids=np.array([39012,39170,39241,39364,39631,40862,42221,43114,43683,44620,46066,46846]).astype(int)

z=np.zeros(size[0])
z[0]=.01
for i in range(len(z)-1):
    step=.01
    z[i+1]=z[i]+step*(1+z[i])

for i in range(len(ids)):
    pofz=pzgrid[ids[i]-1]
    dat=Table([pofz,z],names=['P(z)','z'])
    ascii.write(dat,'Pofz/s%s_pofz.dat' % ids[i])
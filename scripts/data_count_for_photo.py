from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
from vtl.Readfile import Readfile

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_pz.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

specids=np.array([i[12:17] for i in speclist]).astype(int)-1

cat = Table.read(fits.open(
    '../../../Desktop/catalogs_for_CLEAR/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS',
    ignore_missing_end=True)[1])

nm=np.array(cat.colnames)

pidx=[]
for i in range(len(nm)):
    if nm[i][:2]=='f_':
        pidx.append(i)

num_phots=np.zeros(len(specids))

for i in range(len(specids)):
    x=np.zeros(len(pidx))
    for ii in range(len(pidx)):
        x[ii]=cat[specids[i]][pidx[ii]]
    x[x==-99]=0
    num_phots[i]=np.count_nonzero(x)

dat=Table([speclist, num_phots],names=['Filename','NUM_of_photo_points'])
ascii.write(dat,'num_photo.dat')
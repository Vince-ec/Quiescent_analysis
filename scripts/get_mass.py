from vtl.Readfile import Readfile
from spec_id import Get_flux_nocont, Scale_model
from astropy.io import fits,ascii
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from glob import glob
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colors = [(0,i,i,i) for i in np.linspace(0,1,3)]
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

###get list of spectra
specs,rshift,meas,check=np.array(Readfile('speclist_10-7.dat',1,is_float=False))
[rshift]=np.array([rshift]).astype(float)

ids=[U[:6] for U in specs]
IDX=[int(U[1:6]) for U in specs]
speclist=np.array(['spec_stacks_nov29/' + U for U in specs])

fp='/Users/vestrada/Desktop/catalogs_for_CLEAR/'

###########SOUTH############
fastS=fits.open(fp+'goodss_3dhst.v4.1.cats/Fast/goodss_3dhst.v4.1.fout.FITS')[1].data
lmassS=np.array(fastS.field('lmass'))
idS=np.array(fastS.field('id'))

##########NORTH#############
fastN=fits.open(fp+'goodsn_3dhst.v4.1.cats/Fast/goodsn_3dhst.v4.1.fout.FITS')[1].data
lmassN=np.array(fastN.field('lmass'))
idN=np.array(fastN.field('id'))

lmass=np.zeros(len(ids))
for i in range(len(ids)):
    if ids[i][0]=='n':
        lmass[i]=lmassN[IDX[i]-1]

    if ids[i][0]=='s':
        lmass[i]=lmassS[IDX[i]-1]

# print ids
# print speclist
# print lmass
# print rshift

dat=Table([ids,speclist,lmass,rshift],names=['ids','filename','lmass','z'])
print dat
ascii.write(dat,'masslist_dec8.dat')
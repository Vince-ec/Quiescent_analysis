import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Photo import Photo

filenames=glob('../Quiscent_galaxies/extractions_quiescent_mar17/*35774.1D*')
print filenames
rng=np.linspace(8000,11300,340)
flstack=np.zeros(len(rng))
for i in range(len(filenames)-1):
    dat=fits.open(filenames[i+1])
    wv=dat[1].data.field('wave')
    fl=dat[1].data.field('flux')
    plt.plot(wv,fl)
    flinterp=interp1d(wv,fl)
    flstack=flstack+flinterp(rng)
plt.show()

plt.plot(wv,fl)
plt.show()

plt.plot(rng,flstack)
plt.show()

photdat=Photo(rng,flstack,201)


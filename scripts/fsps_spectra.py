import numpy as np
import fsps
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits,ascii
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

sp = fsps.StellarPopulation(imf_type=0, tpagb_norm_type=0, zcontinuous=1, logzsol=np.log10(0.015 / 0.019), sfh=0)

wv, fl = np.array(sp.get_spectrum(tage=1.8))

sp2 = fsps.StellarPopulation(imf_type=1, tpagb_norm_type=1, zcontinuous=1, logzsol=np.log10(0.015 / 0.019), sfh=0,
                             smooth_velocity=False)

# print sp2.libraries()

wv2, fl2 = np.array(sp.get_spectrum(tage=1.8))

print sp.stellar_mass

plt.plot(wv,fl)
plt.plot(wv2,fl2)
# plt.xlim(3250,5350)
plt.show()
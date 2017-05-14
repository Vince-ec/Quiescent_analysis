from vtl.Readfile import Readfile
from spec_id import Error,P, Analyze_Stack,Likelihood_contours
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sea
import numpy as np
from astropy.io import fits
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

dat=fits.open('chidat/s35774_specid_chidata.fits')
# dat=fits.open('chidat/stackfit_tau_test_bc03_chidata.fits')

age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]

# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#          0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
metal = np.array([.0001, .0004, .004, .008, .02])

tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10]

M,A=np.meshgrid(metal,age)

Pr,bfage,bfmetal=Analyze_Stack('chidat/ls10.87_bc03_stackfit_chidata.fits', np.array(tau),metal,age)
onesig,twosig=Likelihood_contours(age,metal,Pr)
levels=np.array([twosig,onesig])

plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
plt.contourf(M,A,Pr,40,cmap=cmap)
plt.xlabel('Z/Z$_\odot$',size=13)
plt.ylabel('Age (Gyrs)',size=13)
plt.axhline(bfage,linestyle='-.',color=sea.color_palette('muted')[2])
plt.axvline(bfmetal,linestyle='-.',color=sea.color_palette('muted')[2])
plt.show()
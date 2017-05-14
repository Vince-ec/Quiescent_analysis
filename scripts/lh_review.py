from spec_id  import Stack_spec_normwmean, Stack_model_normwmean, Scale_model, Identify_stack,\
    Analyze_Stack_avgage, Likelihood_contours, Analyze_Stack, Model_fit_stack_normwmean,\
    Model_fit_stack_MCerr_bestfit_nwmean
from glob import glob
import numpy as np
from scipy.interpolate import interp1d, interp2d
import sympy as sp
import matplotlib.pyplot as plt
from astropy.io import fits,ascii
from vtl.Readfile import Readfile
from astropy.table import Table
import cPickle
import os
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)
cmap = sea.cubehelix_palette(12, start=2, rot=.5, dark=0, light=1.1, as_cmap=True)

"""check quality"""
# lh_list=glob('../poster_plots/*LH*')
#
# keeplist=[]
# for i in range(len(lh_list)):
#     os.system("open " + lh_list[i])
#     qual=int(input('Is this likelihood good? (1 or 0)'))
#     if qual==1:
#         keeplist.append(lh_list[i])
#
# ascii.write([keeplist],'good_lh_list.txt')

"""check list"""

# [glist]=Readfile('good_lh_list.txt',is_float=False)
# print len(glist)
# print glist

"""plot lh"""
metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
        2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

M,A=np.meshgrid(metal,age)

## high mass
Prh,bfageh,bfmetalh=Analyze_Stack_avgage('chidat/gt10.87_fsps_std_10_3250-5500_stackfit_chidata.fits', np.array(tau),metal,age)
onesigh,twosigh=Likelihood_contours(age,metal,Prh)
levelsh=np.array([twosigh,onesigh])

## low mass
Prl,bfagel,bfmetall=Analyze_Stack_avgage('chidat/lt10.87_fsps_std_10_3400-5500_stackfit_chidata.fits', np.array(tau),metal,age)
onesigl,twosigl=Likelihood_contours(age,metal,Prl)
levelsl=np.array([twosigl,onesigl])

plt.contour(M,A,Prh,levelsh,colors='k',linewidths=2)
plt.contourf(M,A,Prh,40,cmap=colmap)
plt.plot(bfmetalh,bfageh,'cp',ms=2,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfageh,np.round(bfmetalh/0.019,2)))
##
plt.contour(M,A,Prl,levelsl,colors='k',linewidths=2)
plt.contourf(M,A,Prl,40,cmap=cmap)
plt.plot(bfmetall,bfagel,'cp',ms=2,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfagel,np.round(bfmetall/0.019,2)))
##
plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.minorticks_on()
plt.xlabel('Metallicity (Z$_\odot$)')
plt.ylabel('Age (Gyrs)')
plt.legend()
plt.show()
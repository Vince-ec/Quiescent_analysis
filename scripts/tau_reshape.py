from vtl.Readfile import Readfile
from spec_id import Error,P,Likelihood_contours,Analyze_Stack,Analyze_Stack_avgage
from scipy.interpolate import interp1d,interp2d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sea
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

# age = np.array([0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0])
# metal = np.array([.0001, .0004, .004, .008, .02, ])
metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
newage=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
        2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])

M,A=np.meshgrid(metal,age)

Pr,exa,exm=Analyze_Stack('chidat/gt10.87_bc03_nwmeannm_stackfit_chidata.fits', tau,metal,age)
onesig,twosig=Likelihood_contours(age,metal,Pr)
levels=np.array([twosig,onesig])
print levels
#
a=[np.trapz(U,metal) for U in Pr]
m=[np.trapz(U,age) for U in Pr.T]

gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])

# levels=np.array([201.66782781 , 908.392327 ])

plt.figure()
gs.update(wspace=0.0,hspace=0.0)
ax=plt.subplot(gs[1,0])
plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
plt.contourf(M,A,Pr,40,cmap=cmap)
plt.xlabel('Z/Z$_\odot$',size=20)
plt.ylabel('Age (Gyrs)',size=20)
plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
            label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
plt.legend(fontsize=15)
plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
           ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
plt.minorticks_on()
plt.xlim(0,.0285)
plt.ylim(0,max(age))
plt.tick_params(axis='both', which='major', labelsize=17)

plt.subplot(gs[1,1])
plt.plot(a,age)
plt.ylim(0,max(age))
plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
plt.yticks([])
plt.xticks([])
#
plt.subplot(gs[0,0])
plt.plot(metal,m)
plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
plt.xlim(0,.0285)
plt.yticks([])
plt.xticks([])
plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
plt.savefig('../research_plots/gt10.87_BC03_nwm_prob.png')

from spec_id import Model_fit_grism,Analyze_grism,Likelihood_contours,Scale_model,Identify_grism
import matplotlib.pyplot as plt
from matplotlib import gridspec
from vtl.Readfile import Readfile
import matplotlib.colors as mcolors
import seaborn as sea
from astropy.io import fits
from scipy.interpolate import interp2d,interp1d
from scipy.ndimage.interpolation import rotate
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colors = [(0,i,i,i) for i in np.linspace(0,1,3)]
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_fsps.dat',1,is_float=False))

age = np.array([0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0])
metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
                  0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
tau = np.array([0, 8.0, 8.15, 8.28, 8.43, 8.57, 8.72, 8.86, 9.0, 9.14, 9.29, 9.43, 9.57, 9.71, 9.86, 10.0])

# Model_fit_grism('../../../Clear_data/Simdata/35774_2d_sim.fits',
#                 tau,metal,age,'35774_2dfit')

"""Likelihood distribution"""
P,Bfa,Bfm=Analyze_grism('chidat/35774_2dfit_chidata.fits',
                   tau,metal,age)

# onesig,twosig=Likelihood_contours(age,metal,P)
# print twosig,onesig

M,A=np.meshgrid(metal,age)

a=np.zeros(len(age))
for i in range(len(age)):
    a[i]=np.trapz(P[i],metal)

m=np.zeros(len(metal))
for i in range(len(metal)):
    m[i]=np.trapz(P.T[i],age)

gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,3])

levels=np.array((2.93614799202, 9.50330962614))

# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,P,levels,colors='k',linewidths=2)
# plt.contourf(M,A,P,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(Bfa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (Bfa,np.round(Bfm/0.019,2)))
# plt.axvline(Bfm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(loc=1,fontsize=15)
# plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
# plt.minorticks_on()
# plt.xlim(0,.0285)
# plt.ylim(0,max(age))
# plt.tick_params(axis='both', which='major', labelsize=17)
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(0,max(age))
# plt.axhline(Bfa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(Bfm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# # plt.savefig('../research_plots/35774_2d_likelihood.png')
#
# """Compare"""
grismdata=fits.open('../../../Clear_data/Simdata/35774_2d_sim.fits')
grismimg=grismdata['DATA'].data
grismerr=grismdata['ERR'].data
grismmodel=grismdata['M0.03_A1.42_T8.0_Z1.2'].data

C=Scale_model(grismimg,grismerr,grismmodel)
#
# fig=plt.figure()
# plt.subplot(311)
# plt.imshow(grismimg,cmap='jet',vmin=-0.1, vmax=0.2)
# plt.subplot(312)
# plt.imshow(C*grismmodel,cmap='jet',vmin=-0.1, vmax=0.2)
# plt.subplot(313)
# plt.imshow(grismimg-C*grismmodel,cmap='jet',vmin=-0.1, vmax=0.2)
# plt.show()

"""proposal plot"""


gs1=gridspec.GridSpec(1,3)
gs1.update(wspace=0.0,top=0.99, bottom=0.8)
gs2=gridspec.GridSpec(1,2)
gs2.update(wspace=0.15,top=0.8, bottom=0.15)

plt.figure()
###
### Data
###
plt.subplot(gs1[0,0])
plt.xticks(())
plt.yticks(())
plt.imshow(grismimg,cmap=cmap,vmin=-0.1, vmax=0.2)
plt.title('Data')
###
### Model
###
plt.subplot(gs1[0,1])
plt.xticks(())
plt.yticks(())
plt.imshow(C*grismmodel,cmap=cmap,vmin=-0.1, vmax=0.2)
plt.title('Model')
###
### Residuals
###
plt.subplot(gs1[0,2])
plt.xticks(())
plt.yticks(())
plt.imshow(grismimg-C*grismmodel,cmap=cmap,vmin=-0.1, vmax=0.2)
plt.title('Residual')
###
### Likelihood
###
plt.subplot(gs2[0,0])
plt.contour(M,A,P,levels,colors='k',linewidths=2)
plt.contourf(M,A,P,40,cmap=cmap)
plt.xlabel('Z/Z$_\odot$',size=13)
plt.ylabel('Age (Gyrs)',size=13)
plt.axhline(Bfa,linestyle='-.',color=sea.color_palette('muted')[2],
            label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (Bfa,np.round(Bfm/0.019,2)))
plt.axvline(Bfm,linestyle='-.',color=sea.color_palette('muted')[2])
plt.legend(loc=1,fontsize=11)
plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
           ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
plt.minorticks_on()
plt.xlim(0,.0285)
plt.ylim(0,max(age))
plt.tick_params(axis='both', which='major', labelsize=13)
###
### Best fits
###
plt.subplot(gs2[0,1])
wv,fl,er=np.array(Readfile('../../Grizlsim/35774_1d.dat',1))
mwv,mfl=np.array(Readfile('../../Grizlsim/bestfit_1d.dat',1))
wv/=2.2
mwv/=2.2
imfl=interp1d(mwv,mfl)(wv)
C=Scale_model(fl,er,imfl)

plt.plot(wv,fl,label='GS3-G102_35774')
plt.plot(wv,C*imfl,
         color=sea.color_palette('muted')[2],label='\nBest fit FSPS model\n Z/Z$_\odot$=1.58, t=1.42 Gyrs')
plt.vlines(4102,0,.3,lw=1,linestyles='-.')
plt.text(4110,0.05,'H$_\delta$')
plt.vlines(4861,0,.7,lw=1,linestyles='-.')
plt.text(4870,0.05,'H$_\\beta$')
plt.vlines(5007,0,.7,lw=1,linestyles='-.')
plt.text(5015,0.05,'[OIII]')
plt.vlines(4358,0,.4,lw=1,linestyles='-.')
plt.text(4370,0.05,'Hg+G')
plt.vlines(3934,0,.35,lw=1,linestyles='-.')
plt.vlines(3963,0,.35,lw=1,linestyles='-.')
plt.text(3880,0.4,'CaII')
plt.vlines(3727,0,.35,lw=1,linestyles='-.')
plt.text(3680,0.4,'[OII]')
plt.legend(loc=2,fontsize=11)
plt.ylim(min(fl),max(fl))
plt.xlabel('Wavelength ($\AA$)',size=13)
plt.ylabel('F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=13)
plt.gcf().subplots_adjust(bottom=0.16)
plt.tick_params(axis='both', which='major', labelsize=13)
###
plt.show()
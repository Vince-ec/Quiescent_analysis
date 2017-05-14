from vtl.Readfile import Readfile
from spec_id import Analyze_Stack_avgage, Stack_spec_normwmean,Stack_model_normwmean, Likelihood_contours,\
    Error,Oldest_galaxy,Gauss_dist
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from glob import glob
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colors = [(0,i,i,i) for i in np.linspace(0,1,3)]
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDA=[]  # all masses in sample
IDL=[]  # low mass sample
IDH=[]  # high mass sample

for i in range(len(ids)):
    if 10.0<=lmass[i] and 1<rshift[i]<1.75:
        IDA.append(i)
    if 10.871>lmass[i] and 1<rshift[i]<1.75:
        IDL.append(i)
    if 10.871<lmass[i] and 1<rshift[i]<1.75:
        IDH.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
     2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

# print min(lmass[IDL]),max(lmass[IDL])
# print min(lmass[IDH]),max(lmass[IDH])
# print np.mean(lmass[IDL])
# print np.mean(lmass[IDH])
# print min(rshift[IDL]),max(rshift[IDL])
# print min(rshift[IDH]),max(rshift[IDH])
# print np.mean(rshift[IDL])
# print np.mean(rshift[IDH])

"""Mass histogram"""
# plt.hist(lmass[IDA])
# plt.axvline(10.87,color='k',alpha=.5,linestyle='--')
# plt.xlabel('log(M/M$_\odot$)',size=30)
# plt.ylabel('N',fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.gcf().subplots_adjust(bottom=0.18)
# # plt.show()
# plt.savefig('../poster_plots/mass_hist_10-12.png')
# plt.close()

"""Redshift histogram"""
# plt.hist(rshift[IDA])
# plt.xlabel('Redshift',size=30)
# plt.ylabel('N',fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.gcf().subplots_adjust(bottom=0.18)
# # plt.show()
# plt.savefig('../poster_plots/redshift_hist_10-12.png')
# plt.close()

"""40597 Example spec 2d and 1d"""
# dat=fits.open('../../../Clear_data/extractions_nov_22/GS3/GS3-G102_40597.2D.fits')
# wv,fl,er=np.array(Readfile('spec_stacks_nov29/s40597_stack.dat'))
#
# sens=dat[10].data
# wave=dat[9].data
# isens=interp1d(wave,sens)
# newsens=isens(wv)
# newsens/=max(newsens)
#
# cmap = sea.cubehelix_palette(12, start=2, rot=.5, dark=0, light=1.1, as_cmap=True)
# gs=gridspec.GridSpec(2,1,height_ratios=[1,5],hspace=0)
# ########
# plt.figure(figsize=(15,5))
# plt.subplot(gs[0])
# plt.imshow(dat[5].data-dat[8].data,cmap=cmap, aspect='auto')
# plt.xticks([],[])
# plt.yticks([],[])
# ##########
# plt.subplot(gs[1])
# plt.errorbar(wv,newsens*fl/1E-18,newsens*er/1E-18,color='#465885',ecolor='#a4699f',
#              fmt='o',ms=3,label='GS3-G102_40597\nz=1.217')
# plt.axvspan(3910*2.217, 3979*2.217, color='k', alpha=.1,zorder=0)
# plt.axvspan(3981*2.217, 4030*2.217, color='k', alpha=.1,zorder=0)
# plt.axvspan(4082*2.217, 4122*2.217, color='k', alpha=.1,zorder=0)
# plt.axvspan(4250*2.217, 4400*2.217, color='k', alpha=.1,zorder=0)
# plt.axvspan(4830*2.217, 4930*2.217, color='k', alpha=.1,zorder=0)
# plt.axvspan(5109*2.217, 5250*2.217, color='k', alpha=.1,zorder=0)
# plt.text(3890*2.217,2.25,'Ca HK',fontsize=18)
# plt.text(4080*2.217,2.25,'H$\delta$',fontsize=18)
# plt.text(4275*2.217,2.25,'H$\gamma$+G',fontsize=18)
# plt.text(4860*2.217,2.25,'H$\\beta$',fontsize=18)
# plt.text(5160*2.217,2.25,'Mg',fontsize=18)
# plt.ylim(0,3.5)
# plt.xlim(wave[0],wave[-1])
# plt.ylabel('F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.legend(loc=2,fontsize=20)
# plt.xlabel('Wavelength ($\AA$)',size=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.gcf().subplots_adjust(bottom=0.2 )
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/1d2dspec_40597.png')
# plt.close()

"""39170 Example spec 2d and 1d"""
# dat=fits.open('../../../Clear_data/extractions_nov_22/ERSPRIME/ERSPRIME-G102_39170.2D.fits')
# wv,fl,er=np.array(Readfile('spec_stacks_nov29/s39170_stack.dat'))
#
# sens=dat[10].data
# wave=dat[9].data
# isens=interp1d(wave,sens)
# newsens=isens(wv)
# newsens/=max(newsens)
#
# cmap = sea.cubehelix_palette(12, start=2, rot=.5, dark=0, light=1.1, as_cmap=True)
# gs=gridspec.GridSpec(2,1,height_ratios=[1,5],hspace=0)
# ########
# plt.figure(figsize=(15,5))
# plt.subplot(gs[0])
# plt.imshow(dat[5].data-dat[8].data,cmap=cmap, aspect='auto')
# plt.xticks([],[])
# plt.yticks([],[])
# ##########
# plt.subplot(gs[1])
# plt.errorbar(wv,newsens*fl/1E-18,newsens*er/1E-18,color='#465885',ecolor='#a4699f',
#              fmt='o',ms=3,label='ERSPRIME-G102_39170\nz=1.022')
# plt.axvspan(3910*2.022, 3979*2.022, color='k', alpha=.1,zorder=0)
# plt.axvspan(3981*2.022, 4030*2.022, color='k', alpha=.1,zorder=0)
# plt.axvspan(4082*2.022, 4122*2.022, color='k', alpha=.1,zorder=0)
# plt.axvspan(4250*2.022, 4400*2.022, color='k', alpha=.1,zorder=0)
# plt.axvspan(4830*2.022, 4930*2.022, color='k', alpha=.1,zorder=0)
# plt.axvspan(5109*2.022, 5250*2.022, color='k', alpha=.1,zorder=0)
# plt.text(3890*2.022,2.25,'Ca HK',fontsize=18)
# plt.text(4080*2.022,2.25,'H$\delta$',fontsize=18)
# plt.text(4270*2.022,2.25,'H$\gamma$+G',fontsize=18)
# plt.text(4850*2.022,2.25,'H$\\beta$',fontsize=18)
# plt.text(5160*2.022,2.25,'Mg',fontsize=18)
# plt.ylim(0,5.2)
# plt.xlim(wave[0],wave[-1])
# plt.ylabel('F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.legend(loc=2,fontsize=20)
# plt.xlabel('Wavelength ($\AA$)',size=30)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/1d2dspec_39170.png')
# plt.close()

""">10.87 stack"""
# wverr,flxerr=np.array(Readfile('flx_err/HM_10_3250-5500.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5500,10))
# ner=np.sqrt(er**2+(fl*flxerr)**2)
# #
# plt.plot(wv,fl*1000,color='#2e4473',label='log($M_*/M_\odot$)>10.87 stack')
# plt.plot(wv,ner*1000,color='#d2686d',label='error')
# # plt.axvspan(3710, 3750, color='k', alpha=.1)
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK', fontsize=10)
# plt.text(4070,7.1,'H$\delta$', fontsize=10)
# plt.text(4248,7.1,'H$\gamma$+G', fontsize=10)
# plt.text(4833,7.1,'H$\\beta$', fontsize=10)
# plt.text(5143,7.1,'Mg', fontsize=10)
# plt.xlim(3250,5500)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.legend(loc=2,fontsize=16)
# # plt.show()
# plt.savefig('../poster_plots/gt10_87_10-12.png')
# plt.close()

"""<10.87 stack"""
# wverr,flxerr=np.array(Readfile('flx_err/LM_10_3400-5250.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDL],rshift[IDL],np.arange(3410,5250,10))
# ner=np.sqrt(er**2+(fl*flxerr[1:])**2)
#
# plt.plot(wv,fl*1000,color='#2e4473',label='log($M_*/M_\odot$)<10.87 stack')
# plt.plot(wv,ner*1000,color='#d2686d',label='error')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK',fontsize=10)
# plt.text(4071,7.1,'H$\delta$',fontsize=10)
# plt.text(4257,7.1,'H$\gamma$+G',fontsize=10)
# plt.text(4844,7.1,'H$\\beta$',fontsize=10)
# plt.text(5144,7.1,'Mg',fontsize=10)
# plt.xlim(3400,5250)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=16)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/lt10_87_10-15.png')
# plt.close()

"""LH >10.87"""
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([24.50862775 , 702.9791765])
# #
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=30.5)
# plt.ylabel('Average Age (Gyrs)',size=30.5)
# plt.plot(exm,exa,'d',color='#81161B',ms=6,
#             label='log($M_*/M_\odot$)>10.87\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' %
#                   (exa,np.round(exm/0.019,2)))
# plt.legend(loc=3,fontsize=17)
# plt.xticks([0,0.00475,0.0095,0.01425,.019]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019])/0.019,2))
# plt.minorticks_on()
# plt.xlim(0,.019)
# plt.ylim(1,4)
# plt.tick_params(axis='both', which='major', labelsize=17)
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(1,4)
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.xlim(0,.019)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.165,left=0.12)
# # plt.show()
# plt.savefig('../poster_plots/gt10.87_LH_10-15.png')
# plt.close()

"""LH <10.87"""
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([37.1786192,258.25862269])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=30.5)
# plt.ylabel('Average Age (Gyrs)',size=30.5)
# plt.plot(exm,exa,'d',color='#81161B',ms=6,
#             label='log($M_*/M_\odot$)<10.87\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' %
#                   (exa,np.round(exm/0.019,2)))
# plt.legend(loc=3,fontsize=15)
# plt.xticks([0,0.00475,0.0095,0.01425,.019]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019])/0.019,2))
# plt.minorticks_on()
# plt.xlim(0,.019)
# plt.ylim(1,4)
# plt.tick_params(axis='both', which='major', labelsize=17)
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(1,4)
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.xlim(0,.019)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.165,left=0.12)
# # plt.show()
# plt.savefig('../poster_plots/lt10.87_LH_10-15.png')
# plt.close()

"""best fit >10.87 stack"""
# wverr,flxerr=np.array(Readfile('flx_err/HM_10_3250-5500.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5500,10))
# ner=np.sqrt(er**2+(fl*flxerr)**2)
#
# flist=[]
# for i in range(len(rshift[IDH])):
#     flist.append('../../../fsps_models_for_fit/models/m0.0096_a2.64_t0_z%s_model.dat' % rshift[IDH][i])
# mwv,mfl,me=Stack_model_normwmean(speclist[IDH],flist, rshift[IDH], np.arange(wv[0],wv[-1]+10,10))
#
# sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
#                 "xtick.minor.size": 5, "ytick.minor.size": 5})
# plt.figure(figsize=(20,5))
# plt.errorbar(wv,fl*1000,ner*1000,color='#2e4473', fmt='o', ms=5)
# plt.plot(mwv,mfl*1000, color='#81161B',
#          label='log($M_*/M_\odot$)>10.87\nBest fit\nZ/Z$_\odot$=0.51,\nt=2.64 Gyrs')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK',fontsize=25)
# plt.text(4070,7.1,'H$\delta$',fontsize=25)
# plt.text(4260,7.1,'H$\gamma$+G',fontsize=25)
# plt.text(4850,7.1,'H$\\beta$',fontsize=25)
# plt.text(5150,7.1,'Mg',fontsize=25)
# plt.xlim(3250,5500)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=38)
# plt.xlabel('Wavelength ($\AA$)',size=38)
# plt.tick_params(axis='both', which='major', labelsize=23)
# plt.gcf().subplots_adjust(bottom=0.235)
# plt.legend(loc=2,fontsize=20)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/gt10_87_b-fit_10-12.png')
# plt.close()

"""best fit <10.87 stack"""
# wverr,flxerr=np.array(Readfile('flx_err/LM_10_3400-5250.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDL],rshift[IDL],np.arange(3410,5250,10))
# ner=np.sqrt(er**2+(fl*flxerr[1:])**2)
#
# flist=[]
# for i in range(len(rshift[IDH])):
#     flist.append('../../../fsps_models_for_fit/models/m0.0085_a2.56_t0_z%s_model.dat' % rshift[IDH][i])
# mwv,mfl,me=Stack_model_normwmean(speclist[IDH],flist, rshift[IDH], np.arange(wv[0],wv[-1]+10,10))
#
# sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
#                 "xtick.minor.size": 5, "ytick.minor.size": 5})
# plt.figure(figsize=(20,5))
# plt.errorbar(wv,fl*1000,ner*1000,color='#2e4473', fmt='o', ms=5)
# plt.plot(mwv,mfl*1000, color='#81161B',
#          label='log($M_*/M_\odot$)<10.87\nBest fit\nZ/Z$_\odot$=0.45,\nt=2.56 Gyrs')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK',fontsize=25)
# plt.text(4070,7.1,'H$\delta$',fontsize=25)
# plt.text(4260,7.1,'H$\gamma$+G',fontsize=25)
# plt.text(4850,7.1,'H$\\beta$',fontsize=25)
# plt.text(5150,7.1,'Mg',fontsize=25)
# plt.xlim(3500,5250)
# plt.ylim(1,7)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=38)
# plt.xlabel('Wavelength ($\AA$)',size=38)
# plt.tick_params(axis='both', which='major', labelsize=23)
# plt.gcf().subplots_adjust(bottom=0.235)
# plt.legend(loc=2,fontsize=20)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/lt10_87_b-fit_10-15.png')
# plt.close()

"""0.25,0.75,1.0 >10.87 stack"""
wverr,flxerr=np.array(Readfile('flx_err/HM_10_3250-5500.dat'))
wv,fl,er=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5500,10))
ner=np.sqrt(er**2+(fl*flxerr)**2)

Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)

print [age[np.argmax(U)] for U in Pr.T]
print metal

flist25=[]
flist50=[]
flist75=[]
flist100=[]
for i in range(len(rshift[IDH])):
    flist25.append('../../../fsps_models_for_fit/models/m0.0049_a3.56_t0_z%s_model.dat' % rshift[IDH][i])
    flist50.append('../../../fsps_models_for_fit/models/m0.0096_a2.64_t0_z%s_model.dat' % rshift[IDH][i])
    flist75.append('../../../fsps_models_for_fit/models/m0.014_a2.26_t0_z%s_model.dat' % rshift[IDH][i])
    flist100.append('../../../fsps_models_for_fit/models/m0.019_a2.11_t0_z%s_model.dat' % rshift[IDH][i])

mwv1,mfl1,mer1=Stack_model_normwmean(speclist[IDH],flist25,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
mwv2,mfl2,mer2=Stack_model_normwmean(speclist[IDH],flist50,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
mwv3,mfl3,mer3=Stack_model_normwmean(speclist[IDH],flist75,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
mwv4,mfl4,mer4=Stack_model_normwmean(speclist[IDH],flist100,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))

# sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
#                 "xtick.minor.size": 5, "ytick.minor.size": 5})
# plt.figure(figsize=(20,5))
# plt.errorbar(wv,fl*1000,ner*1000,color='#2e4473', fmt='o', ms=5,label='log($M_*/M_\odot$)>10.87')
# plt.plot(mwv1,mfl1*1000,color='#388D2F',label='0.25 Z/Z$_\odot$')
# plt.plot(mwv2,mfl2*1000,color='#81161B',label='0.5 Z/Z$_\odot$',zorder=5)
# plt.plot(mwv3,mfl3*1000,color='#FBA6AA',label='0.75 Z/Z$_\odot$')
# plt.plot(mwv4,mfl4*1000,color='#06173B',label='1.0 Z/Z$_\odot$')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK',fontsize=25)
# plt.text(4070,7.1,'H$\delta$',fontsize=25)
# plt.text(4260,7.1,'H$\gamma$+G',fontsize=25)
# plt.text(4850,7.1,'H$\\beta$',fontsize=25)
# plt.text(5150,7.1,'Mg',fontsize=25)
# plt.xlim(3500,5250)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=38)
# plt.xlabel('Wavelength ($\AA$)',size=38)
# plt.tick_params(axis='both', which='major', labelsize=25)
# plt.gcf().subplots_adjust(bottom=0.235)
# plt.legend(loc=4,fontsize=17)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/gt10_87_mcheck_10-15.png')
# plt.close()
# #
res=mfl2*1000
sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
                "xtick.minor.size": 5, "ytick.minor.size": 5})
plt.figure(figsize=(20,5))
stack=plt.errorbar(wv,100*(fl*1000-res)/res,100*ner*1000/res,color='#2e4473', fmt='o', ms=5,label='log($M_*/M_\odot$)>10.87')
plt.plot(mwv1,100*(mfl1*1000-res)/res,color='#388D2F',label='0.25 Z/Z$_\odot$')
plt.plot(mwv2,100*(mfl2*1000-res)/res,color='#81161B',label='0.5 Z/Z$_\odot$',zorder=5)
plt.plot(mwv3,100*(mfl3*1000-res)/res,color='#FBA6AA',label='0.75 Z/Z$_\odot$')
plt.plot(mwv4,100*(mfl4*1000-res)/res,color='#06173B',label='1.0 Z/Z$_\odot$')
# proxies
m25=plt.axhline(y=50,color='#388D2F')
m50=plt.axhline(y=50,color='#81161B')
m75=plt.axhline(y=50,color='#FBA6AA')
m100=plt.axhline(y=50,color='#06173B')
#
plt.axvspan(3910, 3979, color='k', alpha=.1)
plt.axvspan(3981, 4030, color='k', alpha=.1)
plt.axvspan(4082, 4122, color='k', alpha=.1)
plt.axvspan(4250, 4400, color='k', alpha=.1)
plt.axvspan(4830, 4930, color='k', alpha=.1)
plt.axvspan(5109, 5250, color='k', alpha=.1)
plt.text(3870,31,'Ca HK',fontsize=25)
plt.text(4070,31,'H$\delta$',fontsize=25)
plt.text(4260,31,'H$\gamma$+G',fontsize=25)
plt.text(4850,31,'H$\\beta$',fontsize=25)
plt.text(5150,31,'Mg',fontsize=25)
plt.xlim(3500,5250)
plt.ylim(-35,30)
plt.ylabel('Percent Difference',size=37.5)
plt.xlabel('Wavelength ($\AA$)',size=38)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.gcf().subplots_adjust(bottom=0.235)
l1=plt.legend([m25,m50],['0.25 Z/Z$_\odot$','0.5 Z/Z$_\odot$'],loc=1,fontsize=17,bbox_to_anchor=(0.885, 1))
plt.legend([m75,m100,stack],['0.75 Z/Z$_\odot$','1.0 Z/Z$_\odot$','log($M_*/M_\odot$)>10.87'],loc=4,fontsize=17,
           bbox_to_anchor=(0.951, 0.04))
plt.gca().add_artist(l1)
plt.minorticks_on()
# plt.show()
plt.savefig('../poster_plots/gt10_87_resids_10-15.png')
plt.close()

"""0.25,0.75,1.0 <10.87 stack"""
# wverr,flxerr=np.array(Readfile('flx_err/LM_10_3400-5250.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDL],rshift[IDL],np.arange(3410,5250,10))
# ner=np.sqrt(er**2+(fl*flxerr[1:])**2)

# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
#
# print [age[np.argmax(U)] for U in Pr.T]
# print metal

# flist25=[]
# flist50=[]
# flist75=[]
# flist100=[]
# for i in range(len(rshift[IDH])):
#     flist25.append('../../../fsps_models_for_fit/models/m0.0049_a3.45_t0_z%s_model.dat' % rshift[IDH][i])
#     flist50.append('../../../fsps_models_for_fit/models/m0.0085_a2.56_t0_z%s_model.dat' % rshift[IDH][i])
#     flist75.append('../../../fsps_models_for_fit/models/m0.014_a2.2_t0_z%s_model.dat' % rshift[IDH][i])
#     flist100.append('../../../fsps_models_for_fit/models/m0.019_a2.11_t0_z%s_model.dat' % rshift[IDH][i])
#
# mwv1,mfl1,mer1=Stack_model_normwmean(speclist[IDH],flist25,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
# mwv2,mfl2,mer2=Stack_model_normwmean(speclist[IDH],flist50,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
# mwv3,mfl3,mer3=Stack_model_normwmean(speclist[IDH],flist75,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
# mwv4,mfl4,mer4=Stack_model_normwmean(speclist[IDH],flist100,rshift[IDH],np.arange(wv[0],wv[-1]+10,10))
# #
# sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
#                 "xtick.minor.size": 5, "ytick.minor.size": 5})
# plt.figure(figsize=(20,5))
# plt.errorbar(wv,fl*1000,ner*1000,color='#2e4473', fmt='o', ms=5,label='log($M_*/M_\odot$)<10.87')
# plt.plot(mwv1,mfl1*1000,color='#388D2F',label='0.25 Z/Z$_\odot$')
# plt.plot(mwv2,mfl2*1000,color='#81161B',label='0.45 Z/Z$_\odot$',zorder=5)
# plt.plot(mwv3,mfl3*1000,color='#FBA6AA',label='0.75 Z/Z$_\odot$')
# plt.plot(mwv4,mfl4*1000,color='#06173B',label='1.0 Z/Z$_\odot$')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,7.1,'Ca HK',fontsize=25)
# plt.text(4070,7.1,'H$\delta$',fontsize=25)
# plt.text(4260,7.1,'H$\gamma$+G',fontsize=25)
# plt.text(4850,7.1,'H$\\beta$',fontsize=25)
# plt.text(5150,7.1,'Mg',fontsize=25)
# plt.xlim(3500,5250)
# plt.ylim(1,7)
# plt.ylabel('Relative Flux (F$_\lambda$)',size=25)
# plt.xlabel('Wavelength ($\AA$)',size=25)
# plt.tick_params(axis='both', which='major', labelsize=23)
# plt.gcf().subplots_adjust(bottom=0.19)
# plt.legend(loc=4,fontsize=14)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/lt10_87_mcheck_10-15.png')
# plt.close()
# #
# res=mfl2*1000
# sea.set_style( {"xtick.major.size": 8, "ytick.major.size": 8,
#                 "xtick.minor.size": 5, "ytick.minor.size": 5})
# plt.figure(figsize=(20,5))
# plt.errorbar(wv,100*(fl*1000-res)/res,100*ner*1000/res,color='#2e4473', fmt='o', ms=5,label='log($M_*/M_\odot$)<10.87')
# plt.plot(mwv1,100*(mfl1*1000-res)/res,color='#388D2F',label='0.25 Z/Z$_\odot$')
# plt.plot(mwv2,100*(mfl2*1000-res)/res,color='#81161B',label='0.5 Z/Z$_\odot$',zorder=5)
# plt.plot(mwv3,100*(mfl3*1000-res)/res,color='#FBA6AA',label='0.75 Z/Z$_\odot$')
# plt.plot(mwv4,100*(mfl4*1000-res)/res,color='#06173B',label='1.0 Z/Z$_\odot$')
# plt.axvspan(3910, 3979, color='k', alpha=.1)
# plt.axvspan(3981, 4030, color='k', alpha=.1)
# plt.axvspan(4082, 4122, color='k', alpha=.1)
# plt.axvspan(4250, 4400, color='k', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(5109, 5250, color='k', alpha=.1)
# plt.text(3870,81,'Ca HK',fontsize=25)
# plt.text(4070,81,'H$\delta$',fontsize=25)
# plt.text(4260,81,'H$\gamma$+G',fontsize=25)
# plt.text(4850,81,'H$\\beta$',fontsize=25)
# plt.text(5150,81,'Mg',fontsize=25)
# plt.xlim(3500,5250)
# plt.ylim(-70,80)
# plt.ylabel('Percent Difference',size=25)
# plt.xlabel('Wavelength ($\AA$)',size=25)
# plt.tick_params(axis='both', which='major', labelsize=23)
# plt.gcf().subplots_adjust(bottom=0.19)
# plt.legend(loc=4,fontsize=12)
# plt.minorticks_on()
# # plt.show()
# plt.savefig('../poster_plots/lt10_87_resids_10-15.png')
# plt.close()

"""Standardized Residuals"""
# wverr,flxerr=np.array(Readfile('flx_err/HM_10_3250-5250.dat'))
# wv,fl,er=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5250,10))
# ner=np.sqrt(er**2+(fl*flxerr)**2)
#
# flist=[]
# for i in range(len(rshift[IDH])):
#     flist.append('../../../fsps_models_for_fit/models/m0.0096_a2.64_t0_z%s_model.dat' % rshift[IDH][i])
# mwv,mfl,me=Stack_model_normwmean(speclist[IDH],flist, rshift[IDH], np.arange(wv[0],wv[-1]+10,10))
#
# osr=(fl-mfl)/er
# nsr=(fl-mfl)/ner
# rng=np.linspace(-5,5,200)
#
# plt.figure(figsize=(14,5))
# plt.subplot(121)
# plt.hist(osr,15,color='#4e638f',normed=True,label='No flux error')
# plt.plot(rng,Gauss_dist(rng,0,1),color='#838316',label='Normal distribution')
# plt.axvline(np.std(osr),color='k',alpha=.5,linestyle='--',label='$\sigma$=%0.3f' % np.std(osr))
# plt.axvline(-np.std(osr),color='k',alpha=.5,linestyle='--')
# plt.xlabel('$\\frac{Stack-Model}{\sigma}$',size=20)
# plt.ylabel('Normalized units',size=15)
# plt.legend()
# plt.minorticks_on()
# plt.xlim(-5,5)
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.gcf().subplots_adjust(bottom=0.16)
#
# plt.subplot(122)
# plt.hist(nsr,15,color='#162b57',normed=True,label='Flux error added')
# plt.plot(rng,Gauss_dist(rng,0,1),color='#838316',label='Normal distribution')
# plt.axvline(np.std(nsr),color='k',alpha=.5,linestyle='--',label='$\sigma$=%0.3f' % np.std(nsr))
# plt.axvline(-np.std(nsr),color='k',alpha=.5,linestyle='--')
# plt.xlabel('$\\frac{Stack-Model}{\sigma}$',size=20)
# plt.ylabel('Normalized units',size=15)
# plt.legend()
# plt.minorticks_on()
# plt.xlim(-5,5)
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../poster_plots/std_res_10-16.png')
# plt.close()

"""Age v Redshift, Fumagalli 2016 fig 14"""
# Pr,hexa,hexm=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# ah=[np.trapz(U,metal) for U in Pr]
# hel,heh= Error(ah,age)
# print hel,heh
#
# Pr,lexa,lexm=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# al=[np.trapz(U,metal) for U in Pr]
# lel, leh=Error(al,age)
# print lel,leh
#
# print min(rshift[IDL]),max(rshift[IDL]), (min(rshift[IDL])+max(rshift[IDL]))/2
# print min(rshift[IDH]),max(rshift[IDH]), (min(rshift[IDH])+max(rshift[IDH]))/2
#
# fumx,fumy=Readfile('fumagalli_fig14.dat',0)
# d,fverr=Readfile('fumagalli_fig14_verr.dat',0)
# fherr,d=Readfile('fumagalli_fig14_herr.dat',0)
# z=np.linspace(0,2,100)
# ages=[Oldest_galaxy(a) for a in z]
#
# verr=np.zeros(len(fverr))
# herr=np.zeros(len(fherr))
# for i in range(len(verr)):
#     verr[i]=np.abs(fverr[i]-fumy[i])
#     herr[i]=np.abs(fherr[i]-fumx[i])
#
# currentAxis = plt.gca()
#
# plt.plot(z,ages,'k--',alpha=.5)
# # #########fsps
# # ####ls10.87
# plt.errorbar(1.3,lexa,xerr=[[1.3-1.011],[1.63-1.3]],yerr=[[lexa-lel],[leh-lexa]]
#              ,color='#4e638f',alpha=.5,zorder=5,fmt='s',ms=1,label='log($M_*/M_\odot$)<10.87')
# # plt.errorbar(1.3205,lexa,xerr= 0.291,yerr=[[lexa-lel],[leh-lexa]]
# #              , color='w', zorder=4, lw=4,fmt='s', ms=5)
# # currentAxis.add_patch(Rectangle((1.02,1.918),.56,.638,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
# ####gt10.87
# plt.errorbar(1.4,hexa,xerr=[[1.4-1.009],[1.611-1.4]],yerr=[[hexa-hel],[heh-hexa]]
#              ,color='#560004',zorder=5,fmt='s',ms=1,label='log($M_*/M_\odot$)>10.87')
# # plt.errorbar(1.31,hexa,xerr=0.301,yerr=[[hexa-hel],[heh-hexa]]
# #              , color='w', zorder=4, lw=4,fmt='s', ms=5)
# # currentAxis.add_patch(Rectangle((1.01,1.296),.74,.264,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
#
# ##########other points
# plt.errorbar(fumx[0:3],fumy[0:3],xerr=herr[0:3],yerr=verr[0:3]
#              ,color=sea.color_palette('muted')[0],zorder=1,fmt='o',alpha=.5,ms=1,label='Fumagalli+16 BC03')
# plt.errorbar(fumx[3:6],fumy[3:6],xerr=herr[3:6],yerr=verr[3:6]
#              ,color=sea.color_palette('muted')[2],zorder=2,fmt='o',alpha=.5,ms=1,label='Fumagalli+16 FSPS10')
# plt.errorbar(fumx[6:9],fumy[6:9],xerr=herr[6:9],yerr=verr[6:9]
#              ,color=sea.color_palette('muted')[4],zorder=3,fmt='o',alpha=.5,ms=1,label='Fumagalli+16 FSPSC3K')
# plt.plot(fumx[9],fumy[9],'k^',label='Mendel+15')
# plt.plot(fumx[10:14],fumy[10:14],'kp',label='Choi+14')
# plt.plot(fumx[14],fumy[14],'kd',label='Gallazzi+14')
# plt.plot(fumx[15],fumy[15],'k*',label='Whitaker+13')
# plt.axis([0,2,0,8])
# plt.legend(loc=3,fontsize=10)
# plt.xlabel('Redshift',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(1,6.35,'Age of the Universe',rotation=-31)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../poster_plots/agevred_fum14_10-15.png')

"""Split mass Gallazzi 2014 fig 12"""
# Pr,hexa,hexm=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# mh=[np.trapz(U,age) for U in Pr.T]
# hel,heh= Error(mh,metal)
# print hel,heh
# print hel/0.019,heh/0.019
# print np.trapz(np.multiply(mh,metal),metal)/0.019


# Pr,lexa,lexm=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# ml=[np.trapz(U,age) for U in Pr.T]
# lel, leh=Error(ml,metal)
# print lel,leh
# print lel/0.019,leh/0.019
# print np.trapz(np.multiply(mh,metal),metal)/0.019

# print min(lmass[IDL]),max(lmass[IDL])
# print min(lmass[IDH]),max(lmass[IDH])
#
# logm,metal=Readfile('Gallazzi_12.dat',0)
# cvx,cvy=Readfile('Gallazzi_12_line.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
#
# currentAxis = plt.gca()
# ##<10.9
# lm=plt.errorbar(10.48,np.log10(lexm/.019),xerr=[[.39],[.39]],yerr=np.abs([[np.log10(lexm/lel)],
#     [np.log10(lexm/leh)]]),zorder=5,color='#124FD2',ms=1,label='log($M_*/M_\odot$)<10.87')
#
# ###>10.9
# hm=plt.errorbar(11.045,np.log10(hexm/.019),xerr=[[.165],[.165]],yerr=np.abs([[np.log10(hexm/hel)],
#     [np.log10(hexm/heh)]]),zorder=5,color='#E40081',ms=1,label='log($M_*/M_\odot$)>10.87')
# #
# ####gallazzi points
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5, label='SDSS')
# sdss=plt.axhline(y=5,linestyle='--',color='k',alpha=.5)
# plt.plot(cvx,cvy,zorder=2,color='k',alpha=.5, label='Gallazzi+14 best fit line')
# galbf=plt.axhline(y=5,color='k',alpha=.5)
# gal=plt.scatter(logm,metal,zorder=3,color=sea.color_palette('muted')[4],label='Gallazzi+14,z=0.7')
# plt.legend([lm,hm,gal,galbf,sdss],['log($M_*/M_\odot$)<10.87','log($M_*/M_\odot$)>10.87',
#                                     'Gallazzi+14,z=0.7','Gallazzi+14 best fit line','SDSS'],loc=3,fontsize=15)
#
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../poster_plots/massvmetal_gal12_10-16.png')

"""Age v Redshift, Fumagalli 2016 fig 14-rectangle"""
# Pr,hexa,hexm=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# ah=[np.trapz(U,metal) for U in Pr]
# hel,heh= Error(ah,age)
# print hel,heh
# print np.trapz(np.multiply(ah,age),age)
#
# Pr,lexa,lexm=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# al=[np.trapz(U,metal) for U in Pr]
# lel, leh=Error(al,age)
# print lel,leh
# print np.trapz(np.multiply(al,age),age)
#
#
# print min(rshift[IDL]),max(rshift[IDL]), np.mean(rshift[IDL])
# print min(rshift[IDH]),max(rshift[IDH]), np.mean(rshift[IDH])
#
# fumx,fumy=Readfile('fumagalli_fig14.dat',0)
# d,fverr=Readfile('fumagalli_fig14_verr.dat',0)
# fherr,d=Readfile('fumagalli_fig14_herr.dat',0)
# z=np.linspace(0,2,100)
# ages=[Oldest_galaxy(a) for a in z]
#
# verr=np.zeros(len(fverr))
# herr=np.zeros(len(fherr))
# for i in range(len(verr)):
#     verr[i]=np.abs(fverr[i]-fumy[i])
#     herr[i]=np.abs(fherr[i]-fumx[i])
#
# currentAxis = plt.gca()
#
# plt.plot(z,ages,'k--',alpha=.5)
# # #########fsps
# # ####ls10.87
# lme=plt.errorbar(1.191,lexa,xerr=.015,yerr=[[lexa-lel],[leh-lexa]]
#              ,color='w',zorder=5,fmt='s',ms=.1,label='log($M_*/M_\odot$)<10.87')
#
# lm=currentAxis.add_patch(Rectangle((min(rshift[IDL]),lel),max(rshift[IDL])-min(rshift[IDL]),leh-lel
#                                 ,color='#124FD2',zorder=0,alpha=1))
# ####gt10.87
# hme=plt.errorbar(1.31,hexa,xerr=.015,yerr=[[hexa-hel],[heh-hexa]]
#              ,color='w',zorder=5,fmt='s',ms=.1,label='log($M_*/M_\odot$)>10.87')
#
# hm=currentAxis.add_patch(Rectangle((min(rshift[IDH]),hel),max(rshift[IDH])-min(rshift[IDH]),heh-hel
#                                 ,color='#E40081',zorder=0,alpha=1))
# ##########other point
# fumbc=plt.errorbar(fumx[0:3],fumy[0:3],xerr=herr[0:3],yerr=verr[0:3]
#              ,color='#8E9AB3',zorder=1,fmt='o',alpha=1,ms=1)#,label='Fumagalli+16 BC03')
# fumfs=plt.errorbar(fumx[3:6],fumy[3:6],xerr=herr[3:6],yerr=verr[3:6]
#              ,color='#DDF0B8',zorder=2,fmt='o',alpha=1,ms=1)#,label='Fumagalli+16 FSPS10')
# fumck=plt.errorbar(fumx[6:9],fumy[6:9],xerr=herr[6:9],yerr=verr[6:9]
#              ,color='#FFEBC4',zorder=3,fmt='o',alpha=1,ms=1)#,label='Fumagalli+16 FSPSC3K')
# men=plt.scatter(fumx[9],fumy[9],color='k',marker='^',zorder=2)#,label='Mendel+15')
# choi=plt.scatter(fumx[10:14],fumy[10:14],color='k',marker='p',zorder=2)#,label='Choi+14')
# gal=plt.scatter(fumx[14],fumy[14],color='k',marker='d',zorder=2)#,label='Gallazzi+14')
# whit=plt.scatter(fumx[15],fumy[15],color='k',marker='*',zorder=2)#,label='Whitaker+13')
# plt.axis([0,2,0,8])
# l1=plt.legend([men,choi,gal,whit,fumbc,fumfs,fumck],['Mendel+15','Choi+14','Gallazzi+14','Whitaker+13',
#                                                      'Fumagalli+16 BC03','Fumagalli+16 FSPS10','Fumagalli+16 FSPSC3K'],
#                                                         loc=3,fontsize=10)
# plt.legend([(lm,lme),(hm,hme)],['log($M_*/M_\odot$)<10.87','log($M_*/M_\odot$)>10.87'],
#                                                         loc=1,fontsize=14)
# plt.gca().add_artist(l1)
# plt.xlabel('Redshift',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(1,6.35,'Age of the Universe',rotation=-31)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../poster_plots/agevred_fum14_10-16-rec.png')

"""Split mass Gallazzi 2014 fig 12-rectangle"""
# Pr,hexa,hexm=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_3250-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# mh=[np.trapz(U,age) for U in Pr.T]
# hel,heh= Error(mh,metal)
# print hel,heh
#
# Pr,lexa,lexm=Analyze_Stack_avgage('chidat/lt10.87_fsps_10_3400-5250_stackfit_chidata.fits', np.array(tau),metal,age)
# ml=[np.trapz(U,age) for U in Pr.T]
# lel, leh=Error(ml,metal)
# print lel,leh
#
# print min(lmass[IDL]),max(lmass[IDL])
# print min(lmass[IDH]),max(lmass[IDH])
#
# logm,metal=Readfile('Gallazzi_12.dat',0)
# cvx,cvy=Readfile('Gallazzi_12_line.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
#
# currentAxis = plt.gca()
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5, label='SDSS')
# plt.plot(cvx,cvy,zorder=2,color='k',alpha=.5, label='Gallazzi+14 best fit line')
# ##<10.9
# # plt.errorbar(10.48,np.log10(lexm/.019),xerr=[[.39],[.39]],yerr=np.abs([[np.log10(lexm/lel)],
# #     [np.log10(lexm/leh)]]),zorder=5,color='#4e638f',ms=1,label='log($M_*/M_\odot$)<10.87')
#
# currentAxis.add_patch(Rectangle((min(lmass[IDL]),np.log10(lel/.019)),max(lmass[IDL])-min(lmass[IDL]),np.log10(leh/lel),
#                                 color='#4e638f',zorder=5,label='log($M_*/M_\odot$)<10.87'))
#
# ###>10.9
# # plt.errorbar(11.045,np.log10(hexm/.019),xerr=[[.165],[.165]],yerr=np.abs([[np.log10(hexm/hel)],
# #     [np.log10(hexm/heh)]]),zorder=5,color='#560004',ms=1,label='log($M_*/M_\odot$)>10.87')
#
# currentAxis.add_patch(Rectangle((min(lmass[IDH]),np.log10(hel/.019)),max(lmass[IDH])-min(lmass[IDH]),np.log10(heh/hel),
#                                 color='#560004',zorder=5,label='log($M_*/M_\odot$)>10.87'))
# ####gallazzi points
# plt.scatter(logm,metal,zorder=3,color=sea.color_palette('muted')[4],label='Gallazzi+14,z=0.7')
# plt.legend(loc=3,fontsize=15)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../poster_plots/massvmetal_gal12_10-16_rec.png')
from vtl.Readfile import Readfile
from spec_id import P,Error, Stack_spec, Stack_model,Oldest_galaxy, Scale_model, \
    Analyze_Stack, Likelihood_contours, Analyze_Stack_avgage,Stack_spec_normwmean,Stack_model_normwmean
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

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_fsps.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

"""Stack"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
#
# plt.plot(wv,s,'k',alpha=.7,linewidth=1,label='Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.3)
# plt.vlines(4102,0.4,1,lw=1,linestyles='-.')
# plt.text(4110,0.5,'H$_\delta$')
# plt.vlines(4000,1,1.6,lw=1,linestyles='-.')
# plt.text(4010,1.4,'4000 $\AA$ Break')
# plt.vlines(4861,0.4,1,lw=1,linestyles='-.')
# plt.text(4870,0.5,'H$_\\beta$')
# plt.vlines(4358,0.4,.9,lw=1,linestyles='-.')
# plt.text(4370,0.5,'Hg+G')
# plt.vlines(5185,0.4,.9,lw=1,linestyles='-.')
# plt.text(5200,0.5,'Mg')
# plt.vlines(3934,0.4,0.8,lw=1,linestyles='-.')
# plt.vlines(3963,0.4,0.8,lw=1,linestyles='-.')
# plt.text(3700,0.5,'Ca H+K')
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/stack.png')

"""spec 2d and 1d"""
# # dat=fits.open('../../../Clear_data/GS3_extractions_quiescent/GS3-G102_35774.2D.fits')
# dat=fits.open('../../../Clear_data/extractions_quiescent_mar17/GS3-G102_35774.2D.fits')
# wv,fl,er=np.array(Readfile(speclist[3],1))
# sens=dat[10].data
# wave=dat[9].data
# print wave
# isens=interp1d(wave,sens)
# newsens=isens(wv)
# newsens/=max(newsens)
# print zsmax[0]
# gs=gridspec.GridSpec(3,1,height_ratios=[1,4.5,1],hspace=0)
# ########
# plt.figure()
# plt.subplot(gs[0])
# plt.imshow(dat[5].data-dat[8].data,cmap=cmap)
# plt.xticks([],[])
# plt.yticks([],[])
# ##########
# plt.subplot(gs[1])
# plt.fill_between(wv,newsens*(fl-er)/1E-18,newsens*(fl+er)/1E-18,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(wv,newsens*fl/1E-18,label='GS3-G102_35774')
# plt.plot([],[],color=sea.color_palette('muted')[5],alpha=.9,linewidth=5,label='Error')
# plt.vlines(4102*(2.2),0,1.1,lw=1,linestyles='-.')
# plt.text(4110*(2.2),0.2,'H$_\delta$')
# plt.vlines(4861*(2.2),0,2.5,lw=1,linestyles='-.')
# plt.text(4870*(2.2),0.2,'H$_\\beta$')
# plt.vlines(5007*(2.2),0,2.5,lw=1,linestyles='-.')
# plt.text(5015*(2.2),0.2,'[OIII]')
# plt.vlines(4358*(2.2),0,1.5,lw=1,linestyles='-.')
# plt.text(4370*(2.2),0.2,'Hg')
# plt.vlines(3934*(2.2),0,1.5,lw=1,linestyles='-.')
# plt.vlines(3963*(2.2),0,1.5,lw=1,linestyles='-.')
# plt.text(3970*(2.2),1.5,'CaII')
# plt.vlines(3727*(2.2),0,1.5,lw=1,linestyles='-.')
# plt.text(3727*(2.2),1.5,'[OII]')
# plt.xticks([])
# plt.ylim(0,3)
# plt.ylabel('Flux',size=20)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.legend(loc=2,fontsize=13)
# ##########
# plt.subplot(gs[2])
# plt.plot(wv,newsens,label='Sensitivity Function')
# plt.yticks([])
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.ylim(0.1,1.1)
# plt.legend(loc=2,fontsize=13)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/1d2dspec.png')
# ##################

"""FSPS probability"""
# dat=fits.open('chidat/stackfit_tau_test_fsps_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
#
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
# Pt=np.transpose(prob)
#
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
# m/=C0
# a=np.zeros(len(age))
# for i in range(len(age)):
#     a[i]=np.trapz(Pt[i],metal)
# a/=C0
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
# [idmax]=np.argwhere(Pt==np.max(Pt))
# exa,exm=age[idmax[0]],metal[idmax[1]]
#
# levels=np.array([15.3168858165,98.0737912726])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pt,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])

# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/FSPS_prob.png')

"""BC03 probability"""
# dat=fits.open('chidat/stackfit_tau_test_bc03_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# metal = np.array([.0001, .0004, .004, .008, .02])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
#
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
# Pt=np.transpose(prob)
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
# m/=C0
#
# a=np.zeros(len(age))
# for i in range(len(age)):
#     a[i]=np.trapz(Pt[i],metal)
# a/=C0
#
# [idmax]=np.argwhere(Pt==np.max(Pt))
# exa,exm=age[idmax[0]],metal[idmax[1]]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
#
# levels=np.array([6.19728320614,17.730723706])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pt,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.02,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
# plt.ylim(min(age),max(age))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.xlim(0,.03)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(min(age),max(age))
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
#
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(min(metal),max(metal))
# plt.xlim(0,.03)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/BC03_prob.png')

"""Best fit FSPS"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
# flist=[]
# for i in range(len(zbin)):
#     flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t8.0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,label='Stack')
# plt.fill_between(wv,s-e,s+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(fwv,fs,color=sea.color_palette('muted')[2],label='\nFSPS best fit\nZ/Z$_\odot$=0.79, t=1.62 Gyrs, '
#                                                           '$\\tau$=0.1 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=13)
# plt.xticks([])
# plt.tick_params(axis='both', which='major', labelsize=11)
#######
# plt.subplot(gs[1])
# plt.plot(wv,s-fs,color=sea.color_palette('muted')[2],label='Residuals')
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=13)
# plt.ylabel('Relative Flux',size=13)
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.legend(fontsize=12)
# plt.yticks([-.3,-.15,0,.15,.3])
# plt.show()
# plt.savefig('../research_plots/fsps_bestfit.png')

"""Best fit BC03"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
# print zlist
# flist=[]
# for i in range(len(zbin)):
#     flist.append('../../../bc03_models_for_fit/models/m0.008_a2.4_t0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zps,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,label='Stack')
# plt.fill_between(wv,s-e,s+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(fwv,fs,color=sea.color_palette('muted')[2],label='\nBC03 best fit\n Z/Z$_\odot$=0.4, t=2.4 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=13)
# plt.xticks([])
# plt.tick_params(axis='both', which='major', labelsize=11)
#####
# plt.subplot(gs[1])
# plt.plot(wv,s-fs,color=sea.color_palette('muted')[2],label='Residuals')
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=13)
# plt.ylabel('Relative Flux',size=13)
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.legend(fontsize=12)
# plt.yticks([-.3,-.15,0,.15,.3])
# plt.show()
# plt.savefig('../research_plots/bc03_bestfit.png')

"""Best fit Combo"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
# flist=[]
# f1list=[]
# for i in range(len(zbin)):
#     flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t8.0_z%s_model.dat' % zbin[i])
#     f1list.append('../../../bc03_models_for_fit/models/m0.008_a2.74_t8.0_z%s_model.dat' % zbin[i])
#
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
# fwv1,fs1,fe1=Stack_model(f1list,zbin,zcount,np.arange(3250,5550,5))
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,'k',alpha=.6,linewidth=1,label='Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.15)
# plt.plot([],[],'k',alpha=.15,linewidth=5,label='Stack errors')
# plt.plot(fwv,fs,color=sea.color_palette('dark')[2],label='FSPS best fit\nZ/Z$_\odot$=0.79, t=1.62 Gyrs, '
#                                                          '$\\tau$=0 Gyrs')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='BC03 best fit\nZ/Z$_\odot$=0.4, t=2.74 Gyrs,'
#                                                           '$\\tau$=0.1 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=20)
# plt.xticks([])
# plt.tick_params(axis='both', which='major', labelsize=14)
#
# plt.subplot(gs[1])
# p1,=plt.plot(wv,s-fs,color=sea.color_palette('dark')[2],label='FSPS residuals')
# p2,=plt.plot(wv,s-fs1,color=sea.color_palette('dark')[0],label='BC03 residuals')
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.ylabel('Relative Flux',size=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# l1=plt.legend([p1],['FSPS residuals'],loc=1,fontsize=12)
# plt.gca().add_artist(l1)
# plt.legend([p2],['BC03 residuals'],loc=4,fontsize=12)
# plt.ylim(-.45,.45)
# plt.yticks([-.3,-.15,0,.15,.3])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/fsps_bc03_bestfit.png')

"""FSPS 1 sigma check"""
# merr,aerr=np.array(Readfile('stackfit_tau_fsps_1sig.dat',1))
# dat=fits.open('chidat/stackfit_tau_test_fsps_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
#
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
#
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
# m/=C0
#
# Pt=np.transpose(prob)
# a=np.zeros(len(age))
# for i in range(len(age)):
#     a[i]=np.trapz(Pt[i],metal)
# a/=C0
#
# exm=np.trapz(metal*m,metal)
# exa=np.trapz(age*a,age)
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
#
# levels=np.array([15.3168858165,98.0737912726])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels, colors='k',linewidths=2)
# plt.scatter(merr,aerr,s=10,color='c',label='Expectation value of 1 $\sigma$\nperturbations to stack')
# plt.xlabel('Z/Z$_\odot$',size=15)
# plt.ylabel('Age (Gyrs)',size=15)
# plt.legend(fontsize=13)
# plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.ylim(0,max(age))
# plt.xlim(0,.0285)
###
# plt.subplot(gs[1,1])
# plt.hist(aerr,50,orientation='horizontal')
# plt.ylim(0,max(age))
# plt.yticks([])
# plt.xticks([])
####
# plt.subplot(gs[0,0])
# plt.hist(merr,50)
# plt.xlim(min(metal),max(metal))
# plt.yticks([])
# plt.xticks([])
# plt.show()
# plt.savefig('../research_plots/FSPS_1sig.png')

"""FSPS 1 sigma best fit check"""
# def Get_repeats(x,y):
#     z=[x,y]
#     tz=np.transpose(z)
#     size=np.zeros(len(tz))
#     for i in range(len(size)):
#         size[i]=len(np.argwhere(tz==tz[i]))/2
#     size/=5.
#     return size
#
# merr,aerr=np.array(Readfile('stackfit_taubf_fsps_1sig.dat',1))
# dat=fits.open('chidat/stackfit_tau_test_fsps_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
#
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
#
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
# m/=C0
#
# Pt=np.transpose(prob)
# a=np.zeros(len(age))
# for i in range(len(age)):
#     a[i]=np.trapz(Pt[i],metal)
# a/=C0
#
# exm=np.trapz(metal*m,metal)
# exa=np.trapz(age*a,age)
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
#
# s=Get_repeats(merr,aerr)
#
# levels=np.array([15.3168858165,98.0737912726])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels, colors='k',linewidths=2)
# plt.scatter(merr,aerr,s=s,color='c',label='Best fit of 1 $\sigma$\nperturbations to stack')
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.legend(fontsize=15)
# plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.ylim(0,max(age))
# plt.xlim(0,.0285)
###
# plt.subplot(gs[1,1])
# plt.hist(aerr,50,orientation='horizontal')
# plt.ylim(0,max(age))
# plt.yticks([])
# plt.xticks([])
###
# plt.subplot(gs[0,0])
# plt.hist(merr,50)
# plt.xlim(min(metal),max(metal))
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/FSPS_1sig_bf.png')

"""BC03 1 sigma check"""
# merr,aerr=np.array(Readfile('stackfit_tau_bc03_1sig.dat',1))
# dat=fits.open('chidat/stackfit_tau_test_bc03_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# metal = np.array([.0001, .0004, .004, .008, .02])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
# Pt=np.transpose(prob)
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
#
# levels=np.array([6.19728320614,17.730723706])
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels,colors='k',linewidths=2)
# plt.scatter(merr,aerr,color='c',s=10,label='Expectation value of 1 $\sigma$\nperturbations to stack')
# plt.xlabel('Z/Z$_\odot$',size=15)
# plt.ylabel('Age (Gyrs)',size=15)
# plt.legend(fontsize=13,loc=4)
# plt.xlim(0,.05)
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.xlim(0,0.03)
# plt.ylim(min(age),max(age))
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=11)
##
# plt.subplot(gs[1,1])
# plt.hist(aerr,50,orientation='horizontal')
# plt.ylim(min(age),max(age))
# plt.yticks([])
# plt.xticks([])
###
# plt.subplot(gs[0,0])
# plt.hist(merr,50)
# plt.xlim(0,.05)
# plt.yticks([])
# plt.xticks([])
# plt.xlim(0,0.03)
# plt.show()
# plt.savefig('../research_plots/bc03_1sig.png')

"""BC03 1 sigma best fit check"""
# def Get_repeats(x,y):
#     z=[x,y]
#     tz=np.transpose(z)
#     size=np.zeros(len(tz))
#     for i in range(len(size)):
#         size[i]=len(np.argwhere(tz==tz[i]))/2
#     size/=5.
#     return size
#
# merr,aerr=np.array(Readfile('stackfit_taubf_bc03_1sig.dat',1))
# dat=fits.open('chidat/stackfit_tau_test_bc03_chidata.fits')
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# metal = np.array([.0001, .0004, .004, .008, .02])
#
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
# prob = P(chi)
# chigr=[]
# for i in range(len(metal)):
#     acomp=[]
#     for ii in range(len(age)):
#         acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
#     chigr.append(acomp)
# prob=chigr
# Pt=np.transpose(prob)
# M,A=np.meshgrid(metal,age)
#
# m=np.zeros(len(metal))
# for i in range(len(metal)):
#     m[i]=np.trapz(prob[i],age)
# C0=np.trapz(m,metal)
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# Pt/=C0
#
# levels=np.array([6.19728320614,17.730723706])
#
# s=Get_repeats(merr,aerr)
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pt,levels,colors='k',linewidths=2)
# plt.scatter(merr,aerr,color='c',s=s,label='Best fit of 1 $\sigma$\nperturbations to stack')
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.legend(fontsize=15,loc=4)
# plt.xlim(0,.05)
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.xlim(0,0.03)
# plt.ylim(min(age),max(age))
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=17)
#
# plt.subplot(gs[1,1])
# plt.hist(aerr,50,orientation='horizontal')
# plt.ylim(min(age),max(age))
# plt.yticks([])
# plt.xticks([])
#
# plt.subplot(gs[0,0])
# plt.hist(merr,50)
# plt.xlim(0,.05)
# plt.yticks([])
# plt.xticks([])
# plt.xlim(0,0.03)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/bc03_1sig_bf.png')

"""UVJ plot"""
# def Mag(band):
#     magnitude=25-2.5*np.log10(band)
#     return magnitude
#
# fp='/Users/vestrada/Desktop/catalogs_for_CLEAR/'
# fp='/Users/Vince.ec/catalogs_for_CLEAR/'
#
# cat=fits.open(fp+'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS')
# cat_data=cat[1].data
# IDS=np.array(cat_data.field('id'))
# J=np.array(cat_data.field('f_F125W'))
# star=np.array(cat_data.field('class_star'))
#
# fast=fits.open(fp+'goodss_3dhst.v4.1.cats/Fast/goodss_3dhst.v4.1.fout.FITS')
# fast_data=fast[1].data
# lmass=np.array(fast_data.field('lmass'))
#
# restdata=Readfile(fp+'goodss_3dhst.v4.1.cats/RF_colors/goodss_3dhst.v4.1.master.RF',27)
# ids=np.array(restdata[0])
# z=np.array(restdata[1])
# u=np.array(restdata[3])
# v=np.array(restdata[7])
# j=np.array(restdata[9])
#
# ###########Get colors##############
#
# uv=Mag(u)-Mag(v)
# vj=Mag(v)-Mag(j)
#
# IDX=[]
#
# for i in range(len(IDS)):
#     if 1<z[i]<1.75 and star[i]<0.8 and lmass[i] > 10:
#         IDX.append(i)
#
# IDXQ=[]
# IDXsf=[]
#
# for i in IDX:
#     if uv[i]>=0.88*vj[i]+0.59 and uv[i]>1.382 and vj[i]<1.65:
#         IDXQ.append(i)
#     if uv[i]<0.88*vj[i]+0.59:
#         IDXsf.append(i)
#
# plt.scatter(vj[IDXQ],uv[IDXQ],color=sea.color_palette('muted')[2],label='Quiescent')
# plt.scatter(vj[IDXsf],uv[IDXsf],color=sea.color_palette('muted')[0],label='Star Forming')
# plt.plot([0,.9],[1.382,1.382],'k',lw=.9)
# plt.plot([1.65,1.65],[2.05,2.5],'k',lw=.9)
# plt.plot([.9,1.65],[0.88*.9+0.59,0.88*1.65+0.59],'k',lw=.9)
# plt.text(1.8,.75,'Goods South\nlog(M/M$_\odot$)>10\n1<z<1.75',fontsize=17)
# plt.xlabel('Rest-frame V-J (Mag)',size=20)
# plt.ylabel('Rest-frame U-V (Mag)',size=20)
# plt.axis([0,2.5,.5,2.5])
# plt.legend(loc=2,fontsize=17)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/uvj_select.png')

"""Age v Redshift, Fumagalli 2016 fig 14"""
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

# plt.plot(z,ages,'k--',alpha=.5)
##########fsps
# plt.errorbar(1.375,1.62,xerr=.375,yerr=[[1.668-1.37],[1.8-1.62]]
#              ,color=sea.color_palette('dark')[2],zorder=5,fmt='s',ms=9,label='FSPS10')
# plt.errorbar(1.375,1.62,xerr=.375,yerr=[[1.668-1.37],[1.8-1.62]]
#              , color='w', zorder=4, lw=4,fmt='s', ms=10)
#########bc03
# plt.errorbar(1.375,2.74,xerr=.375,yerr=[[2.74-2.21],[3.83-2.74]]
#              ,color=sea.color_palette('dark')[0],zorder=5,fmt='o',ms=9,label='BC03')
# plt.errorbar(1.375,2.74,xerr=.375,yerr=[[2.74-2.21],[3.83-2.74]]
#              , color='w', zorder=4,lw=4, fmt='o', ms=10)
###########other points
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
# plt.legend(loc=3,fontsize=12)
# plt.xlabel('Redshift',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(1,6.35,'Age of the Universe',rotation=-31)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/agevred_fum14.png')
#
# plt.plot(z,ages,'k--',alpha=.5)
###########fsps
# plt.errorbar(1.375, 1.62, xerr=.375, yerr=[[1.668 - 1.37], [1.8 - 1.62]]
#              , color=sea.color_palette('dark')[2], zorder=5, fmt='s', ms=9, label='FSPS10')
# plt.errorbar(1.375, 1.62, xerr=.375, yerr=[[1.668 - 1.37], [1.8 - 1.62]]
#              , color='w', zorder=4, lw=4, fmt='s', ms=10)
#############bc03
# plt.errorbar(1.375, 2.74, xerr=.375, yerr=[[2.74 - 2.21], [3.83 - 2.74]]
#              , color=sea.color_palette('dark')[0], zorder=5, fmt='o', ms=9, label='BC03')
# plt.errorbar(1.375, 2.74, xerr=.375, yerr=[[2.74 - 2.21], [3.83 - 2.74]]
#              , color='w', zorder=4, lw=4, fmt='o', ms=10)
################combined likelihood
# plt.errorbar(1.4,2.59 ,xerr=[[.4],[.35]],yerr=[[2.59-2.36],[2.85-2.59]]
#              ,color=sea.color_palette('dark')[1],zorder=5,fmt='o',ms=9,label='Combined likelihood')
# plt.errorbar(1.4,2.59 ,xerr=[[.4],[.35]],yerr=[[2.59-2.36],[2.85-2.59]]
#              , color='w', zorder=4,lw=4, fmt='o', ms=10)
###############other points
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
# plt.legend(loc=3,fontsize=12)
# plt.xlabel('Redshift',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(1,6.35,'Age of the Universe',rotation=-31)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/agevred_comb_fum14.png')

"""Metallicity v Log mass, Gallazzi 2014 fig 7"""
# logm,metal=Readfile('gallazzi_points.dat',0)
# herr,d1=Readfile('gallazzi_points_herr.dat',0)
# d,verr=Readfile('gallazzi_points_verr.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
# cv2x,cv2y=Readfile('gallazzi_points_curve2.dat',0)
#
# gxerrl,gxerrh,gyerrl,gyerrh=[[],[],[],[]]
# for i in range(len(logm)):
#     gxerrl.append(np.abs(logm[i]-herr[i+i]))
#     gxerrh.append(np.abs(logm[i]-herr[i+i+1]))
#     gyerrl.append(np.abs(metal[i]-verr[i+i+1]))
#     gyerrh.append(np.abs(metal[i]-verr[i+i]))
#
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5)
# plt.plot(cv2x,cv2y,zorder=2,color='k',alpha=.5)
# ####fsps
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9,label='FSPS,1<z<1.75')
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
# #####bc03
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=5,color=sea.color_palette('dark')[0],fmt='o',ms=9,label='BC03,1<z<1.75')
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=4,color='w',lw=4,fmt='o',ms=10)
# ########gallazzi points
# plt.errorbar(logm,metal,xerr=[gxerrl,gxerrh],yerr=[gyerrl,gyerrh],zorder=3,
#              color=sea.color_palette('muted')[4],fmt='o',ms=1,label='Gallazzi+14,z=0.7')
# plt.legend(loc=3,fontsize=15)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(10.02,-.1,'SDSS, z=0',rotation=30)
# plt.text(10.02,-.28,'Gallazzi+14, z=0.7',rotation=30)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/massvmetal_gal.png')

# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5)
# plt.plot(cv2x,cv2y,zorder=2,color='k',alpha=.5)
#####fsps
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9,label='FSPS,1<z<1.75')
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
######bc03
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=5,color=sea.color_palette('dark')[0],fmt='o',ms=9,label='BC03,1<z<1.75')
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=4,color='w',lw=4,fmt='o',ms=10)
########combined likelihoods
# plt.errorbar(10.9,np.log10(0.54),xerr=[[.4],[.3]],yerr=np.abs([[np.log10(.54/.42)],[np.log10(.54/.73)]]),
#              zorder=5,color=sea.color_palette('dark')[1],fmt='o',ms=9,label='Combined likelihoods,1<z<1.75')
# plt.errorbar(10.9,np.log10(0.54),xerr=[[.4],[.3]],yerr=np.abs([[np.log10(.54/.42)],[np.log10(.54/.73)]]),
#              zorder=4,color='w',lw=4,fmt='o',ms=9)
##########gallazzi points
# plt.errorbar(logm,metal,xerr=[gxerrl,gxerrh],yerr=[gyerrl,gyerrh],zorder=3,
#              color=sea.color_palette('muted')[4],fmt='o',ms=1,label='Gallazzi+14,z=0.7')
# plt.legend(loc=3,fontsize=15)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(10.02,-.1,'SDSS, z=0',rotation=30)
# plt.text(10.02,-.28,'Gallazzi+14, z=0.7',rotation=30)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/massvmetal_comb_gal.png')

"""Mass hist"""
# ID,mass,z=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# mass=np.array(mass).astype(float)
# z=np.array(z).astype(float)
#
# print np.average(z)
#
# plt.hist(z,bins=20)
# plt.show()

# lmass=[U for U in mass if U <= 10.87]
# hmass=[U for U in mass if U > 10.87]
#
# print len(lmass), len(hmass)
#
# plt.hist(mass)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('N',fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/mass_hist.png')

"""FSPS 1 sig check"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zlist.append(int(zsmax[i] * 100) / 5 / 20.)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
# flist=[]
# for i in range(len(zbin)):
#     flist.append('../../../fsps_models_for_fit/models/m0.012_a2.11_t8.0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
#
# plt.plot(wv,s,label='Stack')
# plt.fill_between(wv,s-e,s+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(fwv,fs,color=sea.color_palette('muted')[2],label='\nFSPS best fit\nZ/Z$_\odot$=0.63, t=2.11 Gyrs, '
#                                                           '$\\tau$=0.1 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=13)
# plt.xticks([])
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.xlabel('Wavelength ($\AA$)',size=13)
# plt.show()
# plt.savefig('../research_plots/fsps_bestfit.png')

"""Probility compare"""
# dat1=fits.open('chidat/stackfit_tau_test_fsps_chidata.fits')
# dat2=fits.open('chidat/stackfit_tau_test_bc03_chidata.fits')
# age = np.array([0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0])
# metal1 = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# metal2 = np.array([.0001, .0004, .004, .008, .02])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
#
##############probablilties########
# chi1 = []
# for i in range(len(metal1)):
#     chi1.append(dat1[i + 1].data)
# chi1 = np.array(chi1)
# prob1 = P(chi1)
#
# chi2 = []
# for i in range(len(metal2)):
#     chi2.append(dat2[i + 1].data)
# chi2 = np.array(chi2)
# prob2 = P(chi2)
###########integrate tau out#######
# chigr1=[]
# for i in range(len(metal1)):
#     acomp1=[]
#     for ii in range(len(age)):
#         acomp1.append(np.trapz(prob1[i][ii],np.power(10,tau-9)))
#     chigr1.append(acomp1)
# prob1=chigr1
# Pt1=np.transpose(prob1)
#
# chigr2=[]
# for i in range(len(metal2)):
#     acomp2=[]
#     for ii in range(len(age)):
#         acomp2.append(np.trapz(prob2[i][ii],np.power(10,tau-9)))
#     chigr2.append(acomp2)
# prob2=chigr2
# Pt2=np.transpose(prob2)
########metal########
# m1=np.zeros(len(metal1))
# for i in range(len(metal1)):
#     m1[i]=np.trapz(prob1[i],age)
# C1=np.trapz(m1,metal1)
# m1/=C1
#
# m2=np.zeros(len(metal2))
# for i in range(len(metal2)):
#     m2[i]=np.trapz(prob2[i],age)
# C2=np.trapz(m2,metal2)
# m2/=C2
###########age###########
# a1=np.zeros(len(age))
# a2=np.zeros(len(age))
# for i in range(len(age)):
#     a1[i]=np.trapz(Pt1[i],metal1)
#     a2[i]=np.trapz(Pt2[i],metal2)
# a1/=C1
# a2/=C2
#
##########plots#######
# plt.plot(metal1/0.019,m1*0.019,color=sea.color_palette('dark')[2],label='FSPS')
# plt.plot(metal2/.02,m2*0.02,color=sea.color_palette('dark')[0],label='BC03')
# plt.xlabel('Z/Z$_\odot$',size=15)
# plt.ylabel('P(Z)',size=15)
# plt.minorticks_on()
# plt.legend(fontsize=12)
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.show()
# plt.savefig('../research_plots/fsps_bc03_pZ.png')
# plt.close()
#
# plt.plot(age,a1,color=sea.color_palette('dark')[2],label='FSPS')
# plt.plot(age,a2,color=sea.color_palette('dark')[0],label='BC03')
# plt.xlabel('Age (Gyrs)', size=15)
# plt.ylabel('P(t)',size=15)
# plt.minorticks_on()
# plt.legend(fontsize=12)
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.show()
# plt.savefig('../research_plots/fsps_bc03_pt.png')
# plt.close()
#
# metal1/=0.019
# metal2/=0.02
# m1*=0.019
# m2*=0.02
#
# im1=interp1d(metal1,m1)
# im2=interp1d(metal2,m2)
#
# nmetal=np.logspace(np.log10(0.01052632),np.log10(1),100)
#
# cm=im1(nmetal)*im2(nmetal)
# ca=a1*a2
#
# c0=np.trapz(cm,nmetal)
# cm/=c0
# c1=np.trapz(ca,age)
# ca/=c1
#
# print np.trapz(nmetal*cm,nmetal)
# print np.trapz(age*ca,age)
# print Error(cm,nmetal)
# print Error(ca,age)
#
# plt.plot(metal1,m1,color=sea.color_palette('dark')[2],label='FSPS')
# plt.plot(metal2,m2,color=sea.color_palette('dark')[0],label='BC03')
# plt.plot(nmetal,cm,color=sea.color_palette('dark')[1],label='Combined')
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('P(Z)',size=20)
# plt.minorticks_on()
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/combo_pZ.png')
# plt.close()
#
# plt.plot(age,a1,color=sea.color_palette('dark')[2],label='FSPS')
# plt.plot(age,a2,color=sea.color_palette('dark')[0],label='BC03')
# plt.plot(age,ca,color=sea.color_palette('dark')[1],label='Combined')
# plt.xlabel('Age (Gyrs)', size=20)
# plt.ylabel('P(t)',size=20)
# plt.minorticks_on()
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/combo_pt.png')

"""FSPS_chi_by_eye"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
###############models
# f1list=[]
# f2list=[]
# for i in range(len(zbin)):
#     f1list.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t0_z%s_model.dat' % zbin[i])
#     f2list.append('../../../fsps_models_for_fit/models/m0.0096_a2.4_t0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# fwv1,fs1,fe1=Stack_model(f1list,zbin,zcount,np.arange(3250,5550,5))
# fwv2,fs2,fe2=Stack_model(f2list,zbin,zcount,np.arange(3250,5550,5))
#
#########plots
# plt.plot(wv,s,'k',alpha=.6,linewidth=1,label='Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.15)
# plt.plot([],[],'k',alpha=.15,linewidth=5,label='Stack errors')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[2],label='\nFSPS\nZ/Z$_\odot$=0.79, t=1.62 Gyrs,'
#                                                           '$\\tau$=0 Gyrs')
# plt.plot(fwv1, fs2,color=sea.color_palette('husl')[3],label='\nFSPS\nZ/Z$_\odot$=0.51, t=2.4 Gyrs,'
#                                                              '$\\tau$=0 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylim(0.5,1.5)
# plt.ylabel('Relative Flux',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/fsps_eyechi.png')

"""BC03_chi_by_eye"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
#
# ###############models
# f1list=[]
# f2list=[]
# for i in range(len(zbin)):
#     f1list.append('../../../bc03_models_for_fit/models/m0.008_a2.74_t8.0_z%s_model.dat' % zbin[i])
#     f2list.append('../../../bc03_models_for_fit/models/m0.008_a5.26_t8.0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# fwv1,fs1,fe1=Stack_model(f1list,zbin,zcount,np.arange(3250,5550,5))
# fwv2,fs2,fe2=Stack_model(f2list,zbin,zcount,np.arange(3250,5550,5))
#
# ###############plots
# plt.plot(wv,s,'k',alpha=.6,linewidth=1,label='Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.15)
# plt.plot([],[],'k',alpha=.15,linewidth=5,label='Stack errors')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='\nBC03\nZ/Z$_\odot$=0.4, t=2.74 Gyrs,'
#                                                           '$\\tau$=0.1 Gyrs')
# plt.plot(fwv1, fs2,color='#834C24',label='\nBC03\nZ/Z$_\odot$=0.4, t=5.26 Gyrs,'
#                                                              '$\\tau$=0.1 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylim(0.5,1.5)
# plt.ylabel('Relative Flux',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/bc03_eyechi.png')

"""Gallazzi 2014 fig 12"""
# logm,metal=Readfile('Gallazzi_12.dat',0)
# cvx,cvy=Readfile('Gallazzi_12_line.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
#
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5, label='SDSS')
# plt.plot(cvx,cvy,zorder=2,color='k',alpha=.5, label='Gallazzi+14,z=0.7 line fit')
# ###fsps
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9,label='FSPS,1<z<1.75')
# plt.errorbar(10.8,np.log10(.015/.019),xerr=[[.3],[.4]],yerr=np.abs([[np.log10(.015/.0128)],
#     [np.log10(.015/.0179)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
# ####bc03
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=5,color=sea.color_palette('dark')[0],fmt='o',ms=9,label='BC03,1<z<1.75')
# plt.errorbar(10.85,np.log10(.008/.02),xerr=[[.35],[.35]],yerr=np.abs([[np.log10(.008/.005)],
#     [np.log10(.008/.017)]]),zorder=4,color='w',lw=4,fmt='o',ms=10)
# #######gallazzi points
# plt.scatter(logm,metal,zorder=1,color=sea.color_palette('muted')[4],label='Gallazzi+14,z=0.7')
# plt.legend(loc=3,fontsize=15)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/massvmetal_gal12.png')

"""<10.9 FSPS probability"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack('chidat/ls10.87_fsps_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([5.68298695, 26.92653341])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ls10.87_FSPS_prob.png')

""">10.9 FSPS probability"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack('chidat/gt10.87_fsps_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([128.74410133, 514.02975751])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# # plt.plot(M,A,'k.', alpha=1  ,ms=2)
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/gt10.87_FSPS_prob.png')

"""<10.9 BC03 probability"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# metal = np.array([.0001, .0004, .004, .008, .02])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack('chidat/ls10.87_bc03_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([5.22059444, 14.57731201])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.02,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
# plt.ylim(min(age),max(age))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.xlim(0,.03)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(min(age),max(age))
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
#
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(min(metal),max(metal))
# plt.xlim(0,.03)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ls10.87_BC03_prob.png')

""">10.9 BC03 probability"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# metal = np.array([.0001, .0004, .004, .008, .02, .05])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack('chidat/gt10.87_bc03_2_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([1.86482023, 7.31055291])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.02,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(loc=3,fontsize=15)
# plt.ylim(min(age),max(age))
# plt.xticks([0,.005,.01,.015,.02,.025,.03,.05],np.round(np.array([0,.005,.01,.015,.02,.025,.03,.05])/0.02,2))
# plt.xlim(0,.03)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
#
# plt.subplot(gs[1,1])
# plt.plot(a,age)
# plt.ylim(min(age),max(age))
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
#
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(min(metal),max(metal))
# plt.xlim(0,.03)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/gt10.87_BC03_prob.png')

"""<10.9 stack"""
# ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# lmass,rshift=np.array([lmass,rshift]).astype(float)
# nlist=glob('spec_stacks/*')
#
# IDS=[]
#
# for i in range(len(ids)):
#     if lmass[i]<10.87:
#         IDS.append(i)
#
# speclist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(nlist)):
#         if ids[IDS][i]==nlist[ii][12:18]:
#             speclist.append(nlist[ii])
#
# print np.average(rshift[IDS])
#
# metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#          0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]
# age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#      1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]
#
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift[IDS], 2)
# for i in range(len(speczs)):
#     zinput=int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
#
# flist=[]
# f1list=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.0077_a2.74_t8.0_z%s_model.dat' % zlist[i])
#     f1list.append('../../../bc03_models_for_fit/models/m0.008_a2.74_t8.0_z%s_model.dat' % zlist[i])
# print len(nlist),len(flist),len(speczs),len(zlist)
#
#
# wv,s,e=Stack_spec(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv1,fs1,fe1=Stack_model(speclist,f1list, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,'k',alpha=.7,linewidth=1,label='<10.9 Stack')
# plt.plot(fwv,fs,color=sea.color_palette('dark')[2],label='FSPS best fit\nZ/Z$_\odot$=0.41, t=2.4 Gyrs, '
#                                                          '$\\tau$=0.1 Gyrs')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='BC03 best fit\nZ/Z$_\odot$=0.4, t=6.0 Gyrs,'
#                                                           '$\\tau$=0.1 Gyrs')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.3)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.ylim(-.2,2.1)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
#
# plt.subplot(gs[1])
# p1,=plt.plot(wv,s-fs,color=sea.color_palette('dark')[2],label='FSPS residuals')
# p2,=plt.plot(wv,s-fs1,color=sea.color_palette('dark')[0],label='BC03 residuals')
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=13)
# plt.ylabel('Relative Flux',size=13)
# plt.tick_params(axis='both', which='major', labelsize=11)
# l1=plt.legend([p1],['FSPS residuals'],loc=1,fontsize=12)
# plt.gca().add_artist(l1)
# plt.legend([p2],['BC03 residuals'],loc=4,fontsize=12)
#
# plt.yticks([-.3,-.15,0,.15,.3])
# plt.show()
# plt.savefig('../research_plots/ls10.87_stack.png')

""">10.9 stack"""
# ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# lmass,rshift=np.array([lmass,rshift]).astype(float)
# nlist=glob('spec_stacks/*')
#
# IDS=[]
# for i in range(len(ids)):
#     if 10.87<lmass[i]:
#         IDS.append(i)
#
# print np.average(rshift[IDS])
#
# speclist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(nlist)):
#         if ids[IDS][i]==nlist[ii][12:18]:
#             speclist.append(nlist[ii])
#
# metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#          0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]
# age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#      1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]
#
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift[IDS], 2)
# print speczs
# for i in range(len(speczs)):
#     zinput=int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
#
# flist=[]
# f1list=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.012_a1.62_t8.0_z%s_model.dat' % zlist[i])
#     f1list.append('../../../bc03_models_for_fit/models/m0.02_a4.05_t8.0_z%s_model.dat' % zlist[i])
#
# wv,s,e=Stack_spec(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv1,fs1,fe1=Stack_model(speclist,f1list, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# plt.plot(wv,s,'k',alpha=.7,linewidth=1,label='>10.9 Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.3)
# plt.plot(fwv,fs,color=sea.color_palette('dark')[2],label='FSPS best fit\nZ/Z$_\odot$=0.63, t=1.62 Gyrs, '
#                                                          '$\\tau$=0.1 Gyrs')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='BC03 best fit\nZ/Z$_\odot$=1.0, t=4.05 Gyrs,'
#                                                           '$\\tau$=0.1 Gyrs')
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.ylim(-.2,2.1)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4,fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/gt10.87_stack.png')

"""Split mass Gallazzi 2014 fig 12"""
# logm,metal=Readfile('Gallazzi_12.dat',0)
# cvx,cvy=Readfile('Gallazzi_12_line.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
#
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5, label='SDSS')
# plt.plot(cvx,cvy,zorder=2,color='k',alpha=.5, label='Gallazzi+14 best fit line')
# ##<10.9
# plt.errorbar(10.5,np.log10(.0077/.019),xerr=[[.5],[.4]],yerr=np.abs([[np.log10(.0077/.0062)],
#     [np.log10(.0077/.0096)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9,label='FSPS')
# plt.errorbar(10.5,np.log10(.0077/.019),xerr=[[.5],[.4]],yerr=np.abs([[np.log10(.0077/.0062)],
#     [np.log10(.0077/.0096)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
#
# plt.errorbar(10.45,np.log10(.008/.02),xerr=[[.45],[.45]],yerr=np.abs([[np.log10(.008/.0037)],
#     [np.log10(.008/.019)]]),zorder=5,color=sea.color_palette('dark')[0],fmt='o',ms=9,label='BC03')
# plt.errorbar(10.45,np.log10(.008/.02),xerr=[[.45],[.45]],yerr=np.abs([[np.log10(.008/.0037)],
#     [np.log10(.008/.019)]]),zorder=4,color='w',lw=4,fmt='o',ms=10)
# ###>10.9
# plt.errorbar(11.125,np.log10(.012/.019),xerr=[[.225],[.325]],yerr=np.abs([[np.log10(.012/.0103)],
#     [np.log10(.012/.01408)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9)
# plt.errorbar(11.125,np.log10(.012/.019),xerr=[[.225],[.325]],yerr=np.abs([[np.log10(.012/.0103)],
#     [np.log10(.012/.01408)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
#
# plt.errorbar(11.175,np.log10(.02/.02),xerr=[[.275],[.275]],yerr=np.abs([[np.log10(.02/.008)],
#     [np.log10(.02/.05)]]),zorder=5,color=sea.color_palette('dark')[0],fmt='o',ms=9)
# plt.errorbar(11.175,np.log10(.02/.02),xerr=[[.275],[.275]],yerr=np.abs([[np.log10(.02/.008)],
#     [np.log10(.02/.05)]]),zorder=4,color='w',lw=4,fmt='o',ms=10)
# #####gallazzi points
# plt.scatter(logm,metal,zorder=3,color=sea.color_palette('muted')[4],label='Gallazzi+14,z=0.7')
# plt.legend(loc=3,fontsize=15)
# plt.xlabel('log(M/M$_\odot$)',size=20)
# plt.ylabel('log(Z/Z$_\odot$)',size=20)
# plt.axis([10,11.8,-1.5,.5])
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/splitmass_massvmetal_gal12.png')

"""BC03 metallicity compare"""
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
#
# for i in range(len(zsmax)):
#     zinput=int(zsmax[i] * 100) / 5 / 20.
#     if zinput<1:
#         zinput=1.0
#     zlist.append(zinput)
# for i in range(len(bins)):
#     b = []
#     for ii in range(len(zlist)):
#         if bins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
# print zlist
# flist=[]
# for i in range(len(zbin)):
#     flist.append('../../../bc03_models_for_fit/models/m0.008_a2.4_t0_z%s_model.dat' % zbin[i])
#
# flist2=[]
# for i in range(len(zbin)):
#     flist2.append('../../../bc03_models_for_fit/models/m0.02_a2.4_t0_z%s_model.dat' % zbin[i])
#
# wv,s,e=Stack_spec(speclist,zps,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
# fwv2,fs2,fe2=Stack_model(flist2,zbin,zcount,np.arange(3250,5550,5))
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,label='Stack')
# plt.fill_between(wv,s-e,s+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(fwv,fs,color=sea.color_palette('muted')[2],label='\nBC03 best fit\n Z/Z$_\odot$=0.4, t=2.4 Gyrs')
# plt.plot(fwv2,fs2,color=sea.color_palette('muted')[3],label='\nBC03 best fit\n Z/Z$_\odot$=1, t=2.4 Gyrs')
# plt.legend(fontsize=12)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=13)
# plt.xticks([])
# plt.tick_params(axis='both', which='major', labelsize=11)
# ####
# plt.subplot(gs[1])
# plt.plot(wv,s-fs,color=sea.color_palette('muted')[2],label='Residuals')
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=13)
# plt.ylabel('Relative Flux',size=13)
# plt.tick_params(axis='both', which='major', labelsize=11)
# plt.legend(fontsize=12)
# plt.yticks([-.3,-.15,0,.15,.3])
# plt.show()
# plt.savefig('../research_plots/bc03_metal_comp.png')

"""Norm weighted mean"""

""">10.87 fsps likelihood"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
#                   0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/gt10.87_fsps_nwmeannm_stackfit_chidata.fits', tau,metal,age)
# # # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # # levels=np.array([twosig,onesig])
# # # print levels
# #
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([108.40725535, 1212.2568081])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/gt10.87_FSPS_nwmean_prob.png')

"""<10.87 fsps likelihood"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
#                   0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/ls10.87_fsps_nwmeannm_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
# #
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([38.2364537, 458.23051653])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ls10.87_FSPS_nwmean_prob.png')

""">10.87 bc03 likelihood"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([.0001, .0004, .004, .008, .02, ])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/gt10.87_bc03_nwmeannm_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
#
# levels=np.array([46.49560358, 143.9489608])
#
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/gt10.87_BC03_nwmean_prob.png')

"""<10.87 bc03 likelihood"""
# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
# metal = np.array([.0001, .0004, .004, .008, .02, ])
# tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])
# M,A=np.meshgrid(metal,age)
#
# Pr,exa,exm=Analyze_Stack_avgage('chidat/ls10.87_bc03_nwmeannm_stackfit_chidata.fits', tau,metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# # print levels
#
# a=[np.trapz(U,metal) for U in Pr]
# m=[np.trapz(U,age) for U in Pr.T]
#
# gs=gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,4])
# #
# levels=np.array([40.5275994, 133.69929699])
# #
# plt.figure()
# gs.update(wspace=0.0,hspace=0.0)
# ax=plt.subplot(gs[1,0])
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=cmap)
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2],
#             label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (exa,np.round(exm/0.019,2)))
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.legend(fontsize=15)
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
# plt.axhline(exa,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.yticks([])
# plt.xticks([])
# #
# plt.subplot(gs[0,0])
# plt.plot(metal,m)
# plt.axvline(exm,linestyle='-.',color=sea.color_palette('muted')[2])
# plt.xlim(0,.0285)
# plt.yticks([])
# plt.xticks([])
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/ls10.87_BC03_nwmean_prob.png')

""">10.87 Best Fits"""
# ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# lmass,rshift=np.array([lmass,rshift]).astype(float)
# nlist=glob('spec_stacks/*')
#
# IDS=[]
# for i in range(len(ids)):
#     if 10.87<lmass[i] and 1<rshift[i]<1.75:
#         IDS.append(i)
# #
# print np.average(rshift[IDS])
#
# speclist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(nlist)):
#         if ids[IDS][i]==nlist[ii][12:18]:
#             speclist.append(nlist[ii])
#
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift[IDS], 2)
# print speczs
# for i in range(len(speczs)):
#     zinput=int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
#
# flist=[]
# f1list=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t0_z%s_model.dat' % zlist[i])
#     f1list.append('../../../bc03_models_for_fit/models/m0.008_a2.11_t0_z%s_model.dat' % zlist[i])
#
# wv,s,e=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model_normwmean(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv1,fs1,fe1=Stack_model_normwmean(speclist,f1list, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# #
# fs*=1000
# fs1*=1000
# s*=1000
# e*=1000
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,'k',alpha=.7,linewidth=1,label='>10.9 Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.3)
# plt.plot(fwv,fs,color=sea.color_palette('dark')[2],label='FSPS best fit\nZ/Z$_\odot$=0.63, t=1.62 Gyrs')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='BC03 best fit\nZ/Z$_\odot$=1.0, t=4.05 Gyrs')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4,fontsize=15)
#
# plt.subplot(gs[1])
# plt.plot(wv,np.zeros(len(wv)),'k--',alpha=.8)
# p1,=plt.plot(wv,s-fs,color=sea.color_palette('dark')[2],label='FSPS residuals')
# p2,=plt.plot(wv,s-fs1,color=sea.color_palette('dark')[0],label='BC03 residuals')
# plt.fill_between(wv,-e,e,color='k',alpha=.3)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# l1=plt.legend([p1],['FSPS residuals'],loc=3,fontsize=12)
# plt.gca().add_artist(l1)
# plt.legend([p2],['BC03 residuals'],loc=4,fontsize=12)
# plt.ylim(-2.2,2.2)
# plt.yticks([-1,0,1,])
# plt.gcf().subplots_adjust(bottom=0.16)
# # # plt.show()
# plt.savefig('../research_plots/gt10_87_nwmean_stack.png')

"""<10.87 Best Fits"""
# ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# lmass,rshift=np.array([lmass,rshift]).astype(float)
# nlist=glob('spec_stacks/*')
#
# IDS=[]
# for i in range(len(ids)):
#     if 10.87>lmass[i] and 1<rshift[i]<1.75:
#         IDS.append(i)
#
# print np.average(rshift[IDS])
#
# speclist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(nlist)):
#         if ids[IDS][i]==nlist[ii][12:18]:
#             speclist.append(nlist[ii])
#
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift[IDS], 2)
# print speczs
# for i in range(len(speczs)):
#     zinput=int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
#
# flist=[]
# f1list=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.012_a2.11_t0_z%s_model.dat' % zlist[i])
#     f1list.append('../../../bc03_models_for_fit/models/m0.004_a4.62_t0_z%s_model.dat' % zlist[i])
# #
# wv,s,e=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3500,5500,5))
# fwv,fs,fe=Stack_model_normwmean(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv1,fs1,fe1=Stack_model_normwmean(speclist,f1list, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# fs*=1000
# fs1*=1000
# s*=1000
# e*=1000
#
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,s,'k',alpha=.7,linewidth=1,label='<10.87 Stack')
# plt.fill_between(wv,s-e,s+e,color='k',alpha=.3)
# plt.plot(fwv,fs,color=sea.color_palette('dark')[2],label='FSPS best fit\nZ/Z$_\odot$=0.63, t=1.62 Gyrs')
# plt.plot(fwv1,fs1,color=sea.color_palette('dark')[0],label='BC03 best fit\nZ/Z$_\odot$=1.0, t=4.05 Gyrs')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.xticks([])
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4,fontsize=15)
#
# plt.subplot(gs[1])
# plt.plot(wv,np.zeros(len(wv)),'k--',alpha=.8)
# p1,=plt.plot(wv,s-fs,color=sea.color_palette('dark')[2],label='FSPS residuals')
# p2,=plt.plot(wv,s-fs1,color=sea.color_palette('dark')[0],label='BC03 residuals')
# plt.fill_between(wv,-e,e,color='k',alpha=.3)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# l1=plt.legend([p1],['FSPS residuals'],loc=3,fontsize=12)
# plt.gca().add_artist(l1)
# plt.legend([p2],['BC03 residuals'],loc=4,fontsize=12)
# plt.ylim(-2.2,2.2)
# plt.yticks([-1,0,1,])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ls10_87_nwmean_stack.png')

"""Age v Redshift, Fumagalli 2016 fig 14/ nwmean"""
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
# #########fsps
# ####ls10.87
# plt.errorbar(1.3,2.11,xerr=.28,yerr=[[2.11-1.918],[2.556-2.11]]
#              ,color=sea.color_palette('dark')[2],zorder=5,fmt='s',ms=9,label='FSPS10')
# plt.errorbar(1.3,2.11,xerr=.28,yerr=[[1.668-1.37],[1.56-1.62]]
#              , color='w', zorder=4, lw=4,fmt='s', ms=10)
# currentAxis.add_patch(Rectangle((1.02,1.918),.56,.638,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
# ####gt10.87
# plt.errorbar(1.38,1.42,xerr=.37,yerr=[[1.42-1.296],[1.56-1.42]]
#              ,color=sea.color_palette('dark')[2],zorder=5,fmt='s',ms=9)
# plt.errorbar(1.38,1.42,xerr=.37,yerr=[[1.42-1.296],[1.56-1.42]]
#              , color='w', zorder=4, lw=4,fmt='s', ms=10)
# currentAxis.add_patch(Rectangle((1.01,1.296),.74,.264,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
#
# ########bc03
# ####ls10.87
# plt.errorbar(1.3,4.62,xerr=.28,yerr=[[4.62-4.224],[5.095-4.62]]
#              ,color=sea.color_palette('dark')[0],zorder=5,fmt='o',ms=9,label='BC03')
# plt.errorbar(1.3,4.62,xerr=.28,yerr=[[4.62-4.224],[5.095-4.62]]
#              , color='w', zorder=4,lw=4, fmt='o', ms=10)
# currentAxis.add_patch(Rectangle((1.02,4.224),.56,.871,color=sea.color_palette('dark')[0],zorder=5,alpha=.2))
#
# ####gt10.87
# plt.errorbar(1.38,2.11,xerr=.37,yerr=[[2.11-1.926],[2.308-2.11]]
#              ,color=sea.color_palette('dark')[0],zorder=5,fmt='o',ms=9)
# plt.errorbar(1.38,2.11,xerr=.37,yerr=[[2.11-1.926],[2.308-2.11]]
#              , color='w', zorder=4,lw=4, fmt='o', ms=10)
# currentAxis.add_patch(Rectangle((1.01,1.926),.74,.382,color=sea.color_palette('dark')[0],zorder=5,alpha=.2))
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
# plt.legend(loc=3,fontsize=12)
# plt.xlabel('Redshift',size=20)
# plt.ylabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.minorticks_on()
# plt.text(1,6.35,'Age of the Universe',rotation=-31)
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/agevred_fum14_nwmean.png')

"""Split mass Gallazzi 2014 fig 12/ nwmean"""
# logm,metal=Readfile('Gallazzi_12.dat',0)
# cvx,cvy=Readfile('Gallazzi_12_line.dat',0)
# cv1x,cv1y=Readfile('gallazzi_points_curve1.dat',0)
#
# currentAxis = plt.gca()
# plt.plot(cv1x,cv1y,'--',zorder=1,color='k',alpha=.5, label='SDSS')
# plt.plot(cvx,cvy,zorder=2,color='k',alpha=.5, label='Gallazzi+14 best fit line')
# ##<10.9
# plt.errorbar(10.475,np.log10(.012/.019),xerr=[[.385],[.385]],yerr=np.abs([[np.log10(.012/.011)],
#     [np.log10(.012/.0129)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9,label='FSPS')
# plt.errorbar(10.475,np.log10(.012/.019),xerr=[[.385],[.385]],yerr=np.abs([[np.log10(.012/.011)],
#     [np.log10(.012/.0129)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
# currentAxis.add_patch(Rectangle((10.09,np.log10(.011/.019)),.77,0.069,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
#
# ###>10.9
# plt.errorbar(11.045,np.log10(.015/.019),xerr=[[.165],[.165]],yerr=np.abs([[np.log10(.015/.01426)],
#     [np.log10(.015/.016)]]),zorder=5,color=sea.color_palette('dark')[2],fmt='s',ms=9)
# plt.errorbar(11.045,np.log10(.015/.019),xerr=[[.165],[.165]],yerr=np.abs([[np.log10(.015/.01426)],
#     [np.log10(.015/.016)]]),zorder=4,color='w',lw=4,fmt='s',ms=10)
# currentAxis.add_patch(Rectangle((10.88,np.log10(.01426/.019)),.33,0.05,color=sea.color_palette('dark')[2],zorder=5,alpha=.2))
#
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
# plt.savefig('../research_plots/splitmass_massvmetal_gal12_nwmean.png')

"""2d fitting"""

"""2d best fit 35774"""
# wv,fl,er=np.array(Readfile(speclist[3],1))
# wv/=2.2
# mwv,mfl=np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m0.03_a1.42_t8.0_spec.dat',1))
# imfl=interp1d(mwv,mfl)(wv)
#
# C=Scale_model(fl,er,imfl)
#
# plt.plot(wv,fl/1E-18,label='GS3-G102_35774')
# plt.plot(wv,C*imfl/1E-18,
#          color=sea.color_palette('muted')[2],label='\nBest fit FSPS model\n Z/Z$_\odot$=1.58, t=1.42 Gyrs')
# plt.vlines(4102,0,2.25,lw=1,linestyles='-.')
# plt.text(4110,0.2,'H$_\delta$')
# plt.vlines(4861,0,2.5,lw=1,linestyles='-.')
# plt.text(4870,0.2,'H$_\\beta$')
# plt.vlines(5007,0,2.5,lw=1,linestyles='-.')
# plt.text(5015,0.2,'OIII')
# plt.vlines(4358,0,2.25,lw=1,linestyles='-.')
# plt.text(4370,0.2,'Hg+G')
# plt.vlines(3934,0,1.5,lw=1,linestyles='-.')
# plt.vlines(3963,0,1.5,lw=1,linestyles='-.')
# plt.text(3970,0.2,'CaII')
# plt.vlines(3727,0,1.5,lw=1,linestyles='-.')
# plt.text(3727,0.2,'OII')
#
# plt.legend(loc=2,fontsize=13)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.ylabel('F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=20)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.tick_params(axis='both', which='major', labelsize=13)
# # plt.show()
# plt.savefig('../research_plots/2d_35774_bestfit.png')
# ##################
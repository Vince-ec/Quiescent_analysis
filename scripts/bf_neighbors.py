from spec_id  import Analyze_Stack_avgage,Stack_spec_normwmean,Stack_model_normwmean
import seaborn as sea
from glob import glob
import numpy as np
from matplotlib import gridspec
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""Galaxies"""
###get list of spectra
ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)
nlist=glob('spec_stacks/*')

IDS=[]

for i in range(len(ids)):
    if 10.87<lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

speclist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i]==nlist[ii][12:18]:
            speclist.append(nlist[ii])

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput=int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""Best Fit of Neighbors-Metallicity"""
###Create Likelihood
Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_nwmeannm_stackfit_chidata.fits', np.array(tau),metal,age)

# x=np.argwhere(Pr==np.max(Pr))
# print x[0][0],x[0][1]

# x=np.argwhere(Pr.T[6]==np.max(Pr.T[6]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[7]==np.max(Pr.T[7]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[8]==np.max(Pr.T[8]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[9]==np.max(Pr.T[9]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[10]==np.max(Pr.T[10]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[11]==np.max(Pr.T[11]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[12]==np.max(Pr.T[12]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[13]==np.max(Pr.T[13]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[14]==np.max(Pr.T[14]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[15]==np.max(Pr.T[15]))
# print age[x[0][0]]
# x=np.argwhere(Pr.T[16]==np.max(Pr.T[16]))
# print age[x[0][0]]
#
# print Pr.T[6]
# print Pr.T[7]
# print Pr.T[8]
# print Pr.T[9]
# print Pr.T[10]
# print Pr.T[11]
# print Pr.T[12]
# print Pr.T[13]
# print Pr.T[14]
# print Pr.T[15]
# print Pr.T[16]

###Create model Stacks
flist6=[]
flist7=[]
flist8=[]
flist9=[]
flist10=[]
flist11=[]
flist12=[]
flist13=[]
flist14=[]
flist15=[]
flist16=[]
for i in range(len(zlist)):
    flist6.append('../../../fsps_models_for_fit/models/m0.0068_a2.11_t0_z%s_model.dat' % zlist[i])
    flist7.append('../../../fsps_models_for_fit/models/m0.0077_a2.11_t0_z%s_model.dat' % zlist[i])
    flist8.append('../../../fsps_models_for_fit/models/m0.0085_a2.11_t0_z%s_model.dat' % zlist[i])
    flist9.append('../../../fsps_models_for_fit/models/m0.0096_a2.11_t0_z%s_model.dat' % zlist[i])
    flist10.append('../../../fsps_models_for_fit/models/m0.0106_a2.11_t0_z%s_model.dat' % zlist[i])
    flist11.append('../../../fsps_models_for_fit/models/m0.012_a1.42_t0_z%s_model.dat' % zlist[i])
    flist12.append('../../../fsps_models_for_fit/models/m0.0132_a1.42_t0_z%s_model.dat' % zlist[i])
    flist13.append('../../../fsps_models_for_fit/models/m0.014_a1.42_t0_z%s_model.dat' % zlist[i])
    flist14.append('../../../fsps_models_for_fit/models/m0.015_a1.42_t0_z%s_model.dat' % zlist[i])
    flist15.append('../../../fsps_models_for_fit/models/m0.0164_a1.42_t0_z%s_model.dat' % zlist[i])
    flist16.append('../../../fsps_models_for_fit/models/m0.018_a1.42_t0_z%s_model.dat' % zlist[i])

wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3500,5500,5))
fwv6,fs6,fe6=Stack_model_normwmean(speclist,flist6, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv7,fs7,fe7=Stack_model_normwmean(speclist,flist7, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv8,fs8,fe8=Stack_model_normwmean(speclist,flist8, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv9,fs9,fe9=Stack_model_normwmean(speclist,flist9, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv10,fs10,fe10=Stack_model_normwmean(speclist,flist10, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv11,fs11,fe11=Stack_model_normwmean(speclist,flist11, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv12,fs12,fe12=Stack_model_normwmean(speclist,flist12, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv13,fs13,fe13=Stack_model_normwmean(speclist,flist13, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv14,fs14,fe14=Stack_model_normwmean(speclist,flist14, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv15,fs15,fe15=Stack_model_normwmean(speclist,flist15, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv16,fs16,fe16=Stack_model_normwmean(speclist,flist16, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))

# ###get chi square values
chi=np.zeros(11)
chi[0]=np.round(sum(((fl - fs6) / er) ** 2),2)
chi[1]=np.round(sum(((fl - fs7) / er) ** 2),2)
chi[2]=np.round(sum(((fl - fs8) / er) ** 2),2)
chi[3]=np.round(sum(((fl - fs9) / er) ** 2),2)
chi[4]=np.round(sum(((fl - fs10) / er) ** 2),2)
chi[5]=np.round(sum(((fl - fs11) / er) ** 2),2)
chi[6]=np.round(sum(((fl - fs12) / er) ** 2),2)
chi[7]=np.round(sum(((fl - fs13) / er) ** 2),2)
chi[8]=np.round(sum(((fl - fs14) / er) ** 2),2)
chi[9]=np.round(sum(((fl - fs15) / er) ** 2),2)
chi[10]=np.round(sum(((fl - fs16) / er) ** 2),2)

fs6*=1000
fs7*=1000
fs8*=1000
fs9*=1000
fs10*=1000
fs11*=1000
fs12*=1000
fs13*=1000
fs14*=1000
fs15*=1000
fs16*=1000
fl*=1000
er*=1000
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
plt.plot(wv,fl,'k',alpha=.7,linewidth=1,label='>10.87 Stack')
plt.fill_between(wv,fl-er,fl+er,color='k',alpha=.3)
plt.plot(fwv6,fs6, color='#8E2E4E',alpha=.33,label='Z=%s, $\chi^2$=%s' % (np.round((.0068/.019),2), chi[0]))
plt.plot(fwv7,fs7, color='#8E2E4E',alpha=.66,label='Z=%s, $\chi^2$=%s' % (np.round((.0077/.019),2), chi[1]))
plt.plot(fwv8,fs8, color='#8E2E4E',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.0085/.019),2), chi[2]))
plt.plot(fwv9,fs9, color='#264C67',alpha=.25,label='Z=%s, $\chi^2$=%s' % (np.round((.0096/.019),2), chi[3]))
plt.plot(fwv10,fs10, color='#264C67',alpha=.5,label='Z=%s, $\chi^2$=%s' % (np.round((.0106/.019),2), chi[4]))
plt.plot(fwv11,fs11, color='#264C67',alpha=.75,label='Z=%s, $\chi^2$=%s' % (np.round((.012/.019),2), chi[5]))
plt.plot(fwv12,fs12, color='#264C67',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.0132/.019),2), chi[6]))
plt.plot(fwv13,fs13, color='#5F912F',alpha=.33,label='Z=%s, $\chi^2$=%s' % (np.round((.014/.019),2), chi[7]))
plt.plot(fwv14,fs14, 'k', label='Z=%s, $\chi^2$=%s' % (np.round((.015/.019),2), chi[8]))
plt.plot(fwv15,fs15, color='#5F912F',alpha=.66,label='Z=%s, $\chi^2$=%s' % (np.round((.0164/.019),2), chi[9]))
plt.plot(fwv16,fs16, color='#5F912F',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.018/.019),2), chi[10]))
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv),max(wv))
plt.ylim(.5,6.5)
plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
plt.ylabel('Relative Flux',size=15)
plt.title('N=%s' % len(wv))
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4)
plt.show()
#
# plt.subplot(gs[1])
# plt.plot(wv,np.zeros(len(wv)),'k--',alpha=.8)
# plt.plot(wv,fl-fs12,color='#a13535',label='BC03 residuals')
# plt.plot(wv,fl-fs13,color='#8c6ca2',label='BC03 residuals')
# plt.plot(wv,fl-fs14,color='k',label='BC03 residuals')
# plt.plot(wv,fl-fs15,color='#2a812a',label='BC03 residuals')
# plt.plot(wv,fl-fs16,color='#a19f35',label='BC03 residuals')
# plt.fill_between(wv,-er,er,color='k',alpha=.3)
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
# plt.ylim(-1.1,1.1)
# plt.yticks([-1,0,1,])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/metal_neighbor.png')
# plt.close()

"""Best Fit of Neighbors-Age"""
 ###get metallicities
# x=np.argwhere(Pr[6]==np.max(Pr[6]))
# print metal[x[0][0]]
# x=np.argwhere(Pr[7]==np.max(Pr[7]))
# print metal[x[0][0]]
# x=np.argwhere(Pr[8]==np.max(Pr[8]))
# print metal[x[0][0]]
# x=np.argwhere(Pr[9]==np.max(Pr[9]))
# print metal[x[0][0]]
# x=np.argwhere(Pr[10]==np.max(Pr[10]))
# print metal[x[0][0]]
#
###Create model Stacks
# flist6=[]
# flist7=[]
# flist8=[]
# flist9=[]
# flist10=[]
# for i in range(len(zlist)):
#     flist6.append('../../../fsps_models_for_fit/models/m0.03_a1.1_t0_z%s_model.dat' % zlist[i])
#     flist7.append('../../../fsps_models_for_fit/models/m0.027_a1.25_t0_z%s_model.dat' % zlist[i])
#     flist8.append('../../../fsps_models_for_fit/models/m0.015_a1.42_t0_z%s_model.dat' % zlist[i])
#     flist9.append('../../../fsps_models_for_fit/models/m0.0132_a1.62_t0_z%s_model.dat' % zlist[i])
#     flist10.append('../../../fsps_models_for_fit/models/m0.012_a1.85_t0_z%s_model.dat' % zlist[i])
#
# wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv6,fs6,fe6=Stack_model_normwmean(speclist,flist6, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv7,fs7,fe7=Stack_model_normwmean(speclist,flist7, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv8,fs8,fe8=Stack_model_normwmean(speclist,flist8, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv9,fs9,fe9=Stack_model_normwmean(speclist,flist9, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv10,fs10,fe10=Stack_model_normwmean(speclist,flist10, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# ###get chi square values
# chi=np.zeros(5)
# chi[0]=np.round(sum(((fl - fs6) / er) ** 2),2)
# chi[1]=np.round(sum(((fl - fs7) / er) ** 2),2)
# chi[2]=np.round(sum(((fl - fs8) / er) ** 2),2)
# chi[3]=np.round(sum(((fl - fs9) / er) ** 2),2)
# chi[4]=np.round(sum(((fl - fs10) / er) ** 2),2)
#
# fs6*=1000
# fs7*=1000
# fs8*=1000
# fs9*=1000
# fs10*=1000
# fl*=1000
# er*=1000
# gs=gridspec.GridSpec(2,1,height_ratios=[3,1],hspace=0.0)
#
# plt.figure()
# plt.subplot(gs[0])
# plt.plot(wv,fl,label='>10.87 Stack')
# plt.fill_between(wv,fl-er,fl+er,color='k',alpha=.3)
# plt.plot(fwv6,fs6, color='#A13535',label='Z=%s, t=%s, $\chi^2$=%s' % (np.round((.03/.019),2), 1.1,chi[0]))
# plt.plot(fwv7,fs7, color='#8C6CA2', label='Z=%s, t=%s, $\chi^2$=%s' % (np.round((.027/.019),2),1.25, chi[1]))
# plt.plot(fwv8,fs8, 'k', label='Best Fit, $\chi^2$=%s' %  chi[2])
# plt.plot(fwv9,fs9, color='#2A812A', label='Z=%s, t=%s, $\chi^2$=%s' % (np.round((.0132/.019),2),1.62, chi[3]))
# plt.plot(fwv10,fs10, color='#A19F35', label='Z=%s, t=%s, $\chi^2$=%s' % (np.round((.012/.019),2),1.85, chi[4]))
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.ylim(.5,6.5)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4,fontsize=15)
#
# plt.subplot(gs[1])
# plt.plot(wv,np.zeros(len(wv)),'k--',alpha=.8)
# plt.plot(wv,fl-fs6,color='#a13535',label='BC03 residuals')
# plt.plot(wv,fl-fs7,color='#8c6ca2',label='BC03 residuals')
# plt.plot(wv,fl-fs8,color='k',label='BC03 residuals')
# plt.plot(wv,fl-fs9,color='#2a812a',label='BC03 residuals')
# plt.plot(wv,fl-fs10,color='#a19f35',label='BC03 residuals')
# plt.fill_between(wv,-er,er,color='k',alpha=.3)
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
# plt.ylim(-1.1,1.1)
# plt.yticks([-1,0,1,])
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/age_neighbor.png')
# plt.close()
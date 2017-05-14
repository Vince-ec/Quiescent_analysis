from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.interpolate import interp1d
from spec_id import Stack_spec,Stack_model, Scale_model
import cPickle
import seaborn as sea
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in" ,"ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Stack_spec_normwmean(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
    ################

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=1/np.sqrt(np.sum(weigrid[i]))
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_normwmean(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        C=Scale_model(flux,error,iF)
        mflux=C*iF

        # Fl = iF
        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flentry[mask] = ifl(wv_range[mask]) / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    wv = np.array(wv_range)

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=1/np.sqrt(np.sum(weigrid[i]))
    ################

    return wv, stack, err

def Binchi(wv,chi):
    bins=[[3000, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
          [4400, 4830], [4830, 4930], [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 5500]]
    bn=[]
    bnwv=[]
    for i in range(len(bins)):
        b=[]
        w=[]
        for ii in range(len(wv)):
            if bins[i][0]<=wv[ii]<bins[i][1]:
                b.append(chi[ii])
                w.append(wv[ii])
        bn.append(sum(b))
        bnwv.append((bins[i][0]+bins[i][1])/2.)

    return bnwv,bn

bins = [[3000, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
        [4400, 4830], [4830, 4930], [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 5500]]
age = np.array([0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0])
# metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#                   0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
tau = np.array([8.0])
goodfeat=[1,2,4,6,8,10,12]

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

"""Metal Pickling"""
# ################################

# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
#
# for i in range(len(speczs)):
#     zinput = int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
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
# # ##############Stack spectra################
# #
# wv, fl, err = Stack_spec_normwmean(speclist, speczs, np.arange(3000,5500,5))
# #
# # #############Get list of models to fit againts##############
# #
# filepath = '../../../fsps_models_for_fit/models/'
# modellist = []
# for i in range(len(metal)):
#     m = []
#     for ii in range(len(zlist)):
#         m.append(filepath + 'm%s_a1.62_t8.0_z%s_model.dat' % (metal[i], zlist[ii]))
#     modellist.append(m)
#
# print modellist
# #
# # ###############Pickle spectra##################
# #
# pklname = 'metal_wmean_test.pkl'
#
# pklspec = open(pklname, 'wb')
#
# for i in range(len(metal)):
#         mw, mf, me = Stack_model_normwmean(speclist, modellist[i], speczs, zlist,
#                                          np.arange(wv[0], wv[-1] + 5, 5))
#         cPickle.dump(mf, pklspec, protocol=-1)
#
# pklspec.close()
#
# print 'pickle done'

"""Metal Test"""
wv, fl, err = Stack_spec_normwmean(speclist, speczs, np.arange(3000,5500,5))

red=np.zeros(len(metal))
for i in range(len(bins)):
    r=[]
    for ii in range(len(wv)):
        if bins[i][0]<=wv[ii]<bins[i][1]:
            r.append(ii)
    red[i]=len(r)
outspec = open('metal_wmean_test.pkl', 'rb')

mf=[]
chi=np.zeros(len(metal))
chigf=np.zeros(len(metal))
bwv=[]
bchi=[]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim(0,6500)
ax2 = ax1.twinx()
ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)
# ax2.plot(wv,np.ones(len(wv)),'k--',lw=1,alpha=.1,zorder=0)

for i in range(len(metal)):
    mf.append(np.array(cPickle.load(outspec)))
    chiwv=((fl - mf[i]) / err) ** 2
    chi[i]=sum(chiwv)
    bw,bc=Binchi(wv,chiwv)
    chigf[i]=sum(np.array(bc)[goodfeat])
    bwv.append(bw)
    bchi.append(bc)
    ax1.plot(bwv[i],bchi[i],'o',color='#226666', alpha=float(i)/len(metal),ms=6,zorder=1)

[[IDM]]=np.argwhere(chi==np.min(chi))
print metal[IDM]

ax1.plot(bwv[IDM],bchi[IDM],'k*',ms=10,label='Best Fit')
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
ax1.text(3850, 6000,'Ca HK')
ax1.text(3600, 5500,'4000$\AA$\n Break')
ax1.arrow(3600,5500,400,0,head_width=15, head_length=15)
ax1.text(4065, 6000,'H$_\delta$')
ax1.text(4240, 6000,'H$_\gamma$+G')
ax1.text(4830, 6000,'H$_\\beta$')
ax1.text(4950, 6000,'[OIII]')
ax1.text(5140, 6000,'Mg')
ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
ax1.set_ylabel('$\chi ^2$',size=15)
ax2.set_ylabel('Relative Flux',size=15)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax2.tick_params(axis='both', which='major', labelsize=13)
plt.minorticks_on()
plt.gcf().subplots_adjust(bottom=0.16)
ax1.legend(loc=3,fontsize=13)
# plt.show()
plt.savefig('../research_plots/mcomp_chifeat_nwm.png')
plt.close()
# #
# outspec.close()
# #
#
# plt.plot(wv,fl,color='#7887ab')
# # plt.plot(wv,np.ones(len(wv)),'k--',alpha=.2)
# plt.plot(wv,mf[IDM],color='#226666',label='Best Fit Z=%sZ$_\odot$' % np.round(metal[IDM]/.019,2))
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength $\AA$',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/mcomp_bf_nwm.png')
# plt.close()
# #
# featstd=[np.std(np.transpose(bchi)[U])/red[U] for U in range(len(np.transpose(bchi)))]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_ylim(0,33)
# ax2 = ax1.twinx()
# ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)
# # ax2.plot(wv,np.ones(len(wv)),'k--',lw=1,alpha=.1,zorder=0)
# ax1.plot(bwv[0],featstd,'o',ms=6)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# ax1.text(3850, 31,'Ca HK')
# ax1.text(3600, 29,'4000$\AA$\n Break')
# ax1.arrow(3600,29,400,0,head_width=.15, head_length=15)
# ax1.text(4065, 31,'H$_\delta$')
# ax1.text(4240, 31,'H$_\gamma$+G')
# ax1.text(4830, 31,'H$_\\beta$')
# ax1.text(4950, 31,'[OIII]')
# ax1.text(5140, 31,'Mg')
# ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
# ax1.set_ylabel('$\sigma / \lambda$',size=15)
# ax2.set_ylabel('Relative Flux',size=15)
# ax1.tick_params(axis='both', which='major', labelsize=13)
# ax2.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/mcomp_std_nwm.png')
# plt.close()
#
# [[IDr1]]=np.argwhere(chigf==np.min(chigf))
# [[IDr2]]=np.argwhere((chi-chigf)==np.min(chi-chigf))
#
# plt.plot(metal,np.log10(chi),label='Total')
# plt.plot(metal[IDM],np.log10(chi[IDM]),'ko')
# plt.plot(metal,np.log10(chigf),label='Only Spectral Features',color=sea.color_palette('dark')[1])
# plt.plot(metal[IDr1],np.log10(chigf[IDr1]),'ko')
# plt.plot(metal,np.log10(chi-chigf),label='Difference',color=sea.color_palette('dark')[1],alpha=.5)
# plt.plot(metal[IDr2],np.log10((chi-chigf)[IDr2]),'ko',label='Best Fits')
# plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
#            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
# plt.xlabel('Metallicity Z$_\odot$',size=15)
# plt.ylabel('$\chi ^2$',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=13)
# # plt.show()
# plt.savefig('../research_plots/mcomp_chidist_nwm.png')
# plt.close()
#
# outspec = open('metal_wmean_test.pkl', 'rb')
#
# plt.plot(wv,fl,color='#7887ab')
# plt.fill_between(wv,fl-err,fl+err,color='#7887ab',alpha=.3)
# for i in range(len(metal)):
#     mf=np.array(cPickle.load(outspec))
#     plt.plot(wv,mf,alpha=float(i)/len(metal),color='#D7B56E')
# # plt.axvspan(3910, 3979, alpha=.2)
# # plt.axvspan(3981, 4030, alpha=.2)
# # plt.axvspan(4082, 4122, alpha=.2)
# # plt.axvspan(4250, 4400, alpha=.2)
# # plt.axvspan(4830, 4930, alpha=.2)
# # plt.axvspan(4990, 5030, alpha=.2)
# # plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength $\AA$',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/mcomp_mrange_nwm.png')

"""Age Pickling"""
# ################################
# #
# zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift[IDS], 2)
#
# for i in range(len(speczs)):
#     zinput = int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
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
# # ##############Stack spectra################
#
# wv, fl, err = Stack_spec_normwmean(speclist, speczs, np.arange(3000,5500,5))
#
# # #############Get list of models to fit againts##############
#
# filepath = '../../../fsps_models_for_fit/models/'
# modellist = []
# for i in range(len(age)):
#     a = []
#     for ii in range(len(zlist)):
#         a.append(filepath + 'm0.015_a%s_t8.0_z%s_model.dat' % (age[i], zlist[ii]))
#     modellist.append(a)
#
# print modellist
#
# ###############Pickle spectra##################
#
# pklname = 'age_wmean_test.pkl'
#
# pklspec = open(pklname, 'wb')
#
# for i in range(len(age)):
#         mw, mf, me = Stack_model_normwmean(speclist, modellist[i], speczs, zlist,
#                                          np.arange(wv[0], wv[-1] + 5, 5))
#         cPickle.dump(mf, pklspec, protocol=-1)
#
# pklspec.close()
#
# print 'pickle done'

"""Age Test"""
# wv, fl, err = Stack_spec_normwmean(speclist, speczs, np.arange(3000,5500,5))
#
# red=np.zeros(len(age))
# for i in range(len(bins)):
#     r=[]
#     for ii in range(len(wv)):
#         if bins[i][0]<=wv[ii]<bins[i][1]:
#             r.append(ii)
#     red[i]=len(r)
# outspec = open('age_wmean_test.pkl', 'rb')
#
# mf=[]
# chi=np.zeros(len(age))
# chigf=np.zeros(len(age))
# bwv=[]
# bchi=[]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_ylim(0,37000)
# ax2 = ax1.twinx()
# ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)
# # ax2.plot(wv,np.ones(len(wv)),'k--',lw=1,alpha=.1,zorder=0)
#
# for i in range(len(age)):
#     mf.append(np.array(cPickle.load(outspec)))
#     chiwv=((fl - mf[i]) / err) ** 2
#     chi[i]=sum(chiwv)
#     bw,bc=Binchi(wv,chiwv)
#     chigf[i]=sum(np.array(bc)[goodfeat])
#     bwv.append(bw)
#     bchi.append(bc)
#     ax1.plot(bwv[i],bchi[i],'o',color='#226666', alpha=float(i)/len(age),ms=6,zorder=1)
#
# [[IDM]]=np.argwhere(chi==np.min(chi))
# print age[IDM]
#
# ax1.plot(bwv[IDM],bchi[IDM],'k*',zorder=2,ms=10,label='Best Fit')
# # ax1.plot(bwv[IDM-1],bchi[IDM-1],'kp',zorder=1,alpha=.5,ms=10,label='Feature Best Fit')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# ax1.text(3850, 35000,'Ca HK')
# ax1.text(3600, 33000,'4000$\AA$\n Break')
# ax1.arrow(3600,33000,390,0,head_width=15, head_length=15)
# ax1.text(4065, 35000,'H$_\delta$')
# ax1.text(4240, 35000,'H$_\gamma$+G')
# ax1.text(4830, 35000,'H$_\\beta$')
# ax1.text(4950, 35000,'[OIII]')
# ax1.text(5140, 35000,'Mg')
# ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
# ax1.set_ylabel('$\chi ^2$',size=15)
# ax2.set_ylabel('Relative Flux',size=15)
# ax1.tick_params(axis='both', which='major', labelsize=13)
# ax2.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# ax1.legend(loc=3,fontsize=13)
# # plt.show()
# plt.savefig('../research_plots/acomp_chifeat_nwm.png')
# plt.close()
#
# outspec.close()
#
#
# plt.plot(wv,fl,color='#7887ab')
# # plt.plot(wv,np.ones(len(wv)),'k--',alpha=.2)
# plt.plot(wv,mf[IDM],color='#226666',label='Best Fit t=%s Gyrs' % age[IDM])
# # plt.plot(wv,mf[IDM-1],lw=1,color='#403075',label='Best Fit t=1.85 Gyrs')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength $\AA$',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/acomp_bf_nwm.png')
# plt.close()
#
# featstd=[np.std(np.transpose(bchi)[U])/red[U] for U in range(len(np.transpose(bchi)))]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_ylim(0,220)
# ax2 = ax1.twinx()
# ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)
# # ax2.plot(wv,np.ones(len(wv)),'k--',lw=1,alpha=.1,zorder=0)
# ax1.plot(bwv[0],featstd,'o',ms=6)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# ax1.text(3850, 200,'Ca HK')
# ax1.text(3600, 175,'4000$\AA$\n Break')
# ax1.arrow(3600,175,390,0,head_width=.75, head_length=15)
# ax1.text(4065, 200,'H$_\delta$')
# ax1.text(4240, 200,'H$_\gamma$+G')
# ax1.text(4830, 200,'H$_\\beta$')
# ax1.text(4950, 200,'[OIII]')
# ax1.text(5140, 200,'Mg')
# ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
# ax1.set_ylabel('$\sigma / \lambda$',size=15)
# ax2.set_ylabel('Relative Flux',size=15)
# ax1.tick_params(axis='both', which='major', labelsize=13)
# ax2.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/acomp_std_nwm.png')
# plt.close()
# #
# [[IDr1]]=np.argwhere(chigf==np.min(chigf))
# [[IDr2]]=np.argwhere((chi-chigf)==np.min(chi-chigf))
#
# plt.plot(age,np.log10(chi),label='Total')
# plt.plot(age[IDM],np.log10(chi[IDM]),'ko')
# plt.plot(age,np.log10(chigf),label='Only Spectral Features',color=sea.color_palette('dark')[1])
# plt.plot(age[IDr1],np.log10(chigf[IDr1]),'ko')
# plt.plot(age,np.log10(chi-chigf),label='Difference',color=sea.color_palette('dark')[1],alpha=.5)
# plt.plot(age[IDr2],np.log10((chi-chigf)[IDr2]),'ko',label='Best Fits')
# plt.xlabel('Age (Gyrs)',size=15)
# plt.ylabel('log($\chi ^2$)',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=13)
# # plt.show()
# plt.savefig('../research_plots/acomp_chidist_nwm.png')
# plt.close()
#
# outspec = open('age_wmean_test.pkl', 'rb')
#
# plt.plot(wv,fl,color='#7887ab')
# plt.fill_between(wv,fl-err,fl+err,color='#7887ab',alpha=.3)
# for i in range(len(age)):
#     mf=np.array(cPickle.load(outspec))
#     plt.plot(wv,mf,alpha=float(i)/len(age),color='#D7B56E')
# # plt.axvspan(3910, 3979, alpha=.2)
# # plt.axvspan(3981, 4030, alpha=.2)
# # plt.axvspan(4082, 4122, alpha=.2)
# # plt.axvspan(4250, 4400, alpha=.2)
# # plt.axvspan(4830, 4930, alpha=.2)
# # plt.axvspan(4990, 5030, alpha=.2)
# # plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength $\AA$',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=15)
# # plt.show()
# plt.savefig('../research_plots/acomp_mrange_nwm.png')

"""Feature vs full test"""
# wv, fl, err = Stack_spec(speclist, rshift, np.arange(3000,5500,5))
# #
# outspec = open('age_test.pkl', 'rb')
# #
# mf=[]
# #
# for i in range(len(age)):
#     mf.append(np.array(cPickle.load(outspec)))
#
# outspec.close()
#
# print age[10]
#
# totalnum=np.zeros(100)
# featnum=np.zeros(100)
# for i in range(100):
#     chi=np.zeros(len(age))
#     featchi=np.zeros(len(age))
#     tspec = np.array(mf[10]) + np.random.normal(0, 1, len(wv)) * err
#     for ii in range(len(age)):
#         chiwv=((tspec - mf[ii]) / err) ** 2
#         chi[ii]=sum(chiwv)
#         bw,bc=Binchi(wv,chiwv)
#         featchi[ii]=sum(np.array(bc)[goodfeat])
#     IDt=np.argwhere(chi==min(chi))
#     IDf=np.argwhere(featchi==min(featchi))
#     if age[IDt]==age[10]:
#         totalnum[i]=1
#     if age[IDt]==age[10]:
#         featnum[i]=1
#
# print sum(totalnum)
# print sum(featnum)

"""masking region"""
# def Identify_stack_features(fl, err, mfl,mask):
#     ff = np.ma.masked_array(fl, mask)
#     mm = np.ma.masked_array(mfl, mask)
#     ee = np.ma.masked_array(err, mask)
#
#     x = ((ff - mm) / ee) ** 2
#     chi = np.sum(x)
#     return chi
#
# zlist, zbin, zcount, zbins = [[], [], [], np.linspace(1, 1.8, 17)]
# speczs = np.round(rshift, 2)
#
# for i in range(len(speczs)):
#     zinput = int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
# for i in range(len(zbins)):
#     b = []
#     for ii in range(len(zlist)):
#         if zbins[i] == zlist[ii]:
#             b.append(ii)
#     if len(b) > 0:
#         zcount.append(len(b))
# zbin = sorted(set(zlist))
# wv, fl, err = np.array(Stack_spec(speclist, speczs, np.arange(3000,5500,5)))
#
# mask=np.repeat(True,len(wv))
#
# fbins=[[3910, 4030],[4080, 4125],[4250, 4400],[4830, 4930],[4990, 5030],[5110, 5250]]
#
# for i in range(len(fbins)):
#     for ii in range(len(wv)):
#         if fbins[i][0] <= wv[ii] < fbins[i][1]:
#             mask[ii]=False
#
# red=np.zeros(len(metal))
# for i in range(len(bins)):
#     r=[]
#     for ii in range(len(wv)):
#         if bins[i][0]<=wv[ii]<bins[i][1]:
#             r.append(ii)
#     red[i]=len(r)
# outspec = open('metal_test.pkl', 'rb')
#
# mf=[]
# chifo=np.zeros(len(metal))
# chigf=np.zeros(len(metal))
#
# # print m
#
# for i in range(len(metal)):
#     mf.append(np.array(cPickle.load(outspec)))
#     chiwv=((fl - mf[i]) / err) ** 2
#     bw,bc=Binchi(wv,chiwv)
#     chigf[i]=sum(np.array(bc)[goodfeat])
#     # print len(np.array(bc)[goodfeat])
#     chifo[i]=Identify_stack_features(fl,err,mf[i],mask)
#
# [[IDr1]]=np.argwhere(chigf==np.min(chigf))
# [[IDr2]]=np.argwhere(chifo==np.min(chifo))
#
# plt.plot(metal,chigf-chifo,label='Spectral Features',color=sea.color_palette('dark')[1])
# # plt.plot(metal[IDr1],chigf[IDr1],'ko')
# # plt.plot(metal,chifo,label='Only Spectral Features fit',color=sea.color_palette('dark')[2])
# # plt.plot(metal[IDr2],chifo[IDr2],'ko')
# # plt.xticks([0,0.00475,0.0095,0.01425,.019,0.02375,.0285]
# #            ,np.round(np.array([0,0.00475,0.0095,0.01425,.019,0.02375,.0285])/0.019,2))
# plt.xlabel('Metallicity Z$_\odot$',size=15)
# plt.ylabel('$\chi ^2$',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(fontsize=13)
# plt.show()
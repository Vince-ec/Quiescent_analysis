import numpy as np
from spec_id import Stack_spec_normwmean,Stack_model_normwmean,Likelihood_contours,\
    Analyze_Stack_avgage,Identify_stack, Gauss_dist
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Bin_resids(wv,chi):
    bins=[[3250, 3410], [3410, 3570], [3570, 3740], [3740, 3910], [3910, 3980], [3980, 4030], [4030, 4080],
          [4080, 4125], [4125, 4250], [4250, 4400], [4400, 4530], [4530, 4680], [4680, 4830], [4830, 4930],
          [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 5380], [5380, 5500]]
    bsig=[]
    bn=[]
    bnwv=[]
    for i in range(len(bins)):
        b=[]
        w=[]
        for ii in range(len(wv)):
            if bins[i][0]<=wv[ii]<bins[i][1]:
                b.append(chi[ii])
                w.append(wv[ii])
        bn.append(np.mean(b))
        bsig.append(np.std(b))
        bnwv.append((bins[i][0]+bins[i][1])/2.)

    return bnwv,bn,bsig

def Build_Flerr(wv,err_list):
    bins = [[3250, 3410], [3410, 3570], [3570, 3740], [3740, 3910], [3910, 3980], [3980, 4030], [4030, 4080],
            [4080, 4125], [4125, 4250], [4250, 4400], [4400, 4530], [4530, 4680], [4680, 4830], [4830, 4930],
            [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 5380], [5380, 5500]]
    repeats=[]
    for i in range(len(bins)):
        num=[]
        for ii in range(len(wv)):
            if bins[i][0]<=wv[ii]<bins[i][1]:
                num.append(1)
        repeats.append(sum(num))

    flerr=np.array([])
    for i in range(len(repeats)):
        r=np.repeat(err_list[i],repeats[i])
        flerr=np.append(flerr,r)
    return flerr

def Get_chi_lh(fl,mfl,err):
    chi=np.sum(((fl-mfl)/err)**2)
    lh=np.exp(-chi/2)
    return chi,lh

def Model_fit_stack_normwmean_ereg(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, erange, res=5, fsps=False):

    #############Get redshift info###############

    zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
    speczs = np.round(speczs, 2)

    for i in range(len(speczs)):
        zinput = int(speczs[i] * 100) / 5 / 20.
        if zinput < 1:
            zinput = 1.0
        if zinput >1.8:
            zinput = 1.8
        zlist.append(zinput)
    for i in range(len(bins)):
        b = []
        for ii in range(len(zlist)):
            if bins[i] == zlist[ii]:
                b.append(ii)
        if len(b) > 0:
            zcount.append(len(b))
    zbin = sorted(set(zlist))

    ##############Stack spectra################

    wv,fl,er=Stack_spec_normwmean(speclist,speczs,wv_range)
    err=np.sqrt(er**2+(erange*fl)**2)

    #############Prep output file###############

    chifile='chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps==False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m=[]
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    else:
        filepath = '../../../fsps_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname='pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname)==False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+res,res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid=np.zeros([len(metal),len(A),len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii]=Identify_stack(fl,err,mf)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

"""galaxy selection"""
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

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput=int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

"""Stack Galaxies"""
flist=[]
# flist2=[]
for i in range(len(zlist)):
    flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t8.0_z%s_model.dat' % zlist[i])
#     flist2.append('../../../fsps_models_for_fit/models/m0.0077_a2.11_t0_z%s_model.dat' % zlist[i])
#
wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,10))
mwv,mfl,mer=Stack_model_normwmean(speclist,flist,speczs,zlist,np.arange(wv[0],wv[-1]+10,10))
# mwv2,mfl2,mer2=Stack_model_normwmean(speclist,flist2,speczs,zlist,np.arange(wv[0],wv[-1]+10,10))

"""Get standard residuals"""
# erlist=[0.01,  0.03,  0, 0.05,  0, 0, 0, 0, 0.01,  0.02, 0.05,  0,     0.03,  0, 0,     0, 0, 0, 0.07, 0]
# erlist=[0.075, 0.005, 0, 0.02,  0, 0, 0, 0, 0.03,  0.02, 0.005, 0.02,  0.005, 0, 0.005, 0, 0, 0, 0.1,  0.01]
# erlist=[0.065, 0,     0, 0.065, 0, 0, 0, 0, 0.02,  0.02, 0.05,  0,     0.005, 0, 0.01,  0, 0, 0, 0.01, 0.015]
# erlist=[0.065, 0.01,  0, 0.015, 0, 0, 0, 0, 0,     0.02, 0.05,  0.01,  0.015, 0, 0,     0, 0, 0, 0,    0.1]
# erlist=[0.075, 0.02,  0, 0.045, 0, 0, 0, 0, 0.015, 0.02, 0.025, 0.005, 0.005, 0, 0.015, 0, 0, 0, 0.04, 0.01]
# erlist=[0.055, 0.01,  0, 0.04,  0, 0, 0, 0, 0.015, 0.02, 0.035, 0.005, 0.01,  0, 0.005, 0, 0, 0, 0.04, 0.025]
# erlist=[0.065, 0.01,  0, 0.04,  0, 0, 0, 0, 0.015, 0.02, 0.05,  0.005, 0.005, 0, 0.005, 0, 0, 0, 0.04, 0.001]
erlist=[0.06, 0.08,  0., 0.07,  0, 0, 0, 0, 0.005, 0.03, 0.005, 0, 0.015, 0, 0.1, 0., 0., 0., 0.01, 0.]
flxerr=Build_Flerr(wv,erlist)
#
# error=np.sqrt(er**2+(flxerr*fl)**2)
#
# osr=(fl-mfl)/er
# nsr=(fl-mfl)/error
#
# bwv,binerr,bsig=Bin_resids(wv,osr)

#
# plt.errorbar(wv,fl,error,fmt='o',ms=4)
# plt.plot(mwv,mfl)
# plt.minorticks_on()
# plt.show()
# rng=np.linspace(-3,3,100)

# plt.hist(osr,15,normed=True)
# plt.plot(rng,Gauss_dist(rng,0,1),label='$\mu$=0, $\sigma$=1')
# plt.axvline(np.mean(osr),color='k',alpha=.5,linestyle='-.',label='$\mu$=%0.3f' % np.mean(osr))
# plt.axvline(np.std(osr),color='k',alpha=.5,linestyle='--',label='$\sigma$=%0.3f' % np.std(osr))
# plt.axvline(-np.std(osr),color='k',alpha=.5,linestyle='--')
# plt.xlabel('Sigma',size=15)
# plt.ylabel('Normalized units',size=15)
# plt.legend()
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/org_SR_hist.png')
# plt.close()

# plt.hist(nsr,15,normed=True)
# plt.plot(rng,Gauss_dist(rng,0,1),label='$\mu$=0, $\sigma$=1')
# plt.axvline(np.mean(nsr),color='k',alpha=.5,linestyle='-.',label='$\mu$=%0.3f' % np.mean(nsr))
# plt.axvline(np.std(nsr),color='k',alpha=.5,linestyle='--',label='$\sigma$=%0.3f' % np.std(nsr))
# plt.axvline(-np.std(nsr),color='k',alpha=.5,linestyle='--')
# plt.xlabel('Sigma',size=15)
# plt.ylabel('Normalized units',size=15)
# plt.legend()
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/setB_SR_hist.png')
# plt.close()

# plt.plot(wv,osr,'o')
# plt.errorbar(bwv,binerr,bsig,fmt='o',color=sea.color_palette('dark')[2])
# plt.hlines(1,min(wv),max(wv))
# plt.hlines(-1,min(wv),max(wv))
# plt.hlines(0,min(wv),max(wv),linestyle='--')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(3250,5500)
# plt.ylim(-3,4)
# plt.minorticks_on()
# plt.show()


"""model fit"""
M,A=np.meshgrid(metal,age)

Model_fit_stack_normwmean_ereg(speclist,tau,metal,age,speczs,np.arange(3250,5500,10),
                         'gt10.87_lres10_nerrreg6_stackfit','gt10.87_min_w10lres_spec',flxerr ,res=10,fsps=True)

Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_lres10_nerrreg6_stackfit_chidata.fits', np.array(tau),metal,age)
onesig,twosig=Likelihood_contours(age,metal,Pr)
levels=np.array([twosig,onesig])
print levels
plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
plt.contourf(M,A,Pr,40,cmap=colmap)
plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.minorticks_on()
plt.xlabel('Metallicity (Z$_\odot$)')
plt.ylabel('Age (Gyrs)')
plt.legend()
# plt.show()
plt.savefig('../research_plots/gt10.87_lres10_nerrreg6_lh.png')

"""find best set"""
# perc=np.linspace(.01,.1,10)
# elist=np.zeros(20)
# print len(elist)
#
# mu=np.zeros(len(perc))
# sig=np.zeros(len(perc))
#
# for i in range(len(perc)):
#     elist[19]=perc[i]
#     flxerr = Build_Flerr(wv, elist)
#     error = np.sqrt(er ** 2 + (flxerr * fl) ** 2)
#     sr = (fl - mfl) / error
#     mu[i]=np.mean(sr)
#     sig[i]=np.std(sr)
#
# plt.plot(perc,mu)
# plt.savefig('../research_plots/mu_trend.png')
# plt.close()
#
# plt.plot(perc,sig)
# plt.savefig('../research_plots/sig_trend.png')
# plt.close()

# elist=np.zeros(20)
# slist=np.zeros([100000,len(elist)])
# mu=np.zeros(100000)
# sig=np.zeros(100000)
# ch=np.linspace(0,.1,21)
#
# for i in range(len(mu)):
#     elist[0]=np.random.choice(ch)
#     elist[1]=np.random.choice(ch)
#     elist[3]=np.random.choice(ch)
#     elist[8]=np.random.choice(ch)
#     # elist[9]=np.random.choice(ch)
#     elist[9]=0.03
#     elist[10]=np.random.choice(ch)
#     elist[11]=np.random.choice(ch)
#     elist[12]=np.random.choice(ch)
#     elist[14]=np.random.choice(ch)
#     elist[18]=np.random.choice(ch)
#     elist[19]=np.random.choice(ch)
#     flxerr = Build_Flerr(wv, elist)
#     error = np.sqrt(er ** 2 + (flxerr * fl) ** 2)
#     sr = (fl - mfl) / error
#     mu[i]=np.mean(sr)
#     sig[i]=np.std(sr)
#     slist[i]=elist
#
# MU=np.abs(mu)
# Sig=np.abs(sig-1)
#
# IDX=np.argwhere((MU+Sig)==np.min(MU+Sig))
#
# print mu[IDX]
# print sig[IDX]
# print np.round(slist[IDX],4)
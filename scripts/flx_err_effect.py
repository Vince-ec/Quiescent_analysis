import numpy as np
from spec_id import Gauss_dist, Stack_spec_normwmean, Stack_model_normwmean, Likelihood_contours,\
    Analyze_Stack_avgage,Identify_stack
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Build_Flerr(wv,err_list):
    bins=[[2000, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
          [4400, 4830], [4830, 4930], [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 6000]]
    repeats=[]
    for i in range(len(bins)):
        num=[]
        for ii in range(len(wv)):
            if bins[i][0]<=wv[ii]<bins[i][1]:
                num.append(1)
        repeats.append(sum(num))
        # print repeats[i]


    flerr=np.array([])
    for i in range(len(repeats)):
        r=np.repeat(err_list[i],repeats[i])
        flerr=np.append(flerr,r)
    return flerr

def Model_fit_stack_normwmean_ereg(speclist, tau, metal, A, speczs, wv_range, name, pkl_name, erange, res=5, fsps=False):
    ##############Stack spectra################
    wv, fl, er = Stack_spec_normwmean(speclist, speczs, wv_range)
    err = np.sqrt(er ** 2 + (erange * fl) ** 2)

    #############Prep output file###############

    chifile = 'chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps == False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
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
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname = 'pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist, modellist[i][ii][iii], speczs,
                                                       np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid = np.zeros([len(metal), len(A), len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii] = Identify_stack(fl, err, mf)
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

def Model_fit_stack_MCerr_bestfit_nwmean(speclist, tau, metal, A, speczs, wv_range, name, pklname, erange, repeats=100):
    ##############Stack spectra################
    wv, flx, er = Stack_spec_normwmean(speclist, speczs, wv_range)
    err = np.sqrt(er ** 2 + (erange * flx) ** 2)

    ##############Start loop and add error#############

    mlist = []
    alist = []

    for i in range(repeats):

        outspec = open(pklname, 'rb')

        fl = flx + np.random.normal(0, err)

        ##############Create chigrid#################

        chigrid = np.zeros([len(metal), len(A), len(tau)])
        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mf = np.array(cPickle.load(outspec))
                    chigrid[i][ii][iii] = Identify_stack(fl, err, mf)

        chigrid = np.array(chigrid, dtype=np.float128)
        chi = np.transpose(chigrid)
        ################Find best fit##################

        scale = Readfile('tau_scale_nage.dat', 1)

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        newchi = []
        for i in range(len(chi)):
            if i == 0:
                iframe = chi[i]
            else:
                iframe = interp2d(metal, scale[i], chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newchi.append(iframe)
        newchi = np.transpose(newchi)

        prob = np.exp(-newchi / 2)

        tau = np.array(tau)
        chigr = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(prob[i][ii], np.power(10, tau - 9)))
            chigr.append(acomp)
        prob = np.array(chigr)

        [idmax] = np.argwhere(prob == np.max(prob))
        alist.append(A[idmax[1]])
        mlist.append(metal[idmax[0]])

        outspec.close()

    fn = name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871<lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#      1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
newage=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64,
        2.68, 2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""get flx err array"""
# flist=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.015_a1.42_t0_z%s_model.dat' % zlist[i])
#
w_rng=np.arange(3250,5500,12)
#
wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],w_rng)
# mwv,mfl,mer=Stack_model_normwmean(speclist,flist,speczs,zlist,np.arange(wv[0],wv[-1]+10,10))
#
# elist=[.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]
# slist=[]
# mu=[]
# sig=[]
# ch=np.linspace(0,.1,21)
# for i in range(len(ch)):
#     elist[0] = ch[i]
#     for ii in range(len(ch)):
#         elist[6] = ch[ii]
#         for iii in range(len(ch)):
#             elist[7] = ch[iii]
#             flxerr = Build_Flerr(wv, elist)
#             error = np.sqrt(er ** 2 + (flxerr * fl) ** 2)
#             sr = (fl - mfl) / error
#             mu.append(np.mean(sr))
#             sig.append(np.std(sr))
#             slist.append(np.array(elist))
#
# MU=np.abs(mu)
# Sig=np.abs(np.array(sig)-1)
#
# IDX=np.argwhere((MU+Sig)==np.min(MU+Sig))
#
# print IDX[0][0]
# print mu[IDX[0][0]]
# print sig[IDX[0][0]]
# print slist[IDX[0][0]]
#
# erlist=slist[IDX[0][0]]
flxerr=np.array(Readfile('flx_err_in2.dat'))

"""plot hist"""
# flxerr=Build_Flerr(wv,erlist)
# error=np.sqrt(er**2+(flxerr*fl)**2)
# nsr=(fl-mfl)/error
# rng=np.linspace(-3,3,100)
#
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
# # plt.show()
# plt.savefig('../research_plots/flxer_0-6-7_hist.png')
# plt.close()

"""fit with errlist"""
# mlist=glob('sim_specs/*')
#
# simlist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(mlist)):
#         if ids[IDS][i]==mlist[ii][10:16]:
#             simlist.append(mlist[ii])
#
M,A=np.meshgrid(metal,newage)
#
# Model_fit_stack_normwmean_ereg(simlist,tau,metal,age,rshift[IDS],w_rng,
#                          'flxer_0-6-7_sim_stackfit','sim_3205_5500_spec',flxerr ,res=10,fsps=True)
#
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/flxer_0-6-7_sim_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# plt.savefig('../research_plots/flxer_0-6-7_sim_lh.png')
# plt.close()

Model_fit_stack_MCerr_bestfit_nwmean(speclist[IDS],np.array(tau),metal,newage,rshift[IDS],np.arange(3250,5500,12),
                                     'gt10.87_fsps_nage_flxer_mcerr','pickled_mstacks/gt10.87_nage_flxer_spec.pkl',
                                     flxerr,repeats=1000)

# Model_fit_stack_normwmean_ereg(speclist[IDS],tau,metal,newage,rshift[IDS],w_rng,
#                          'gt10.87_fsps_nage_noer_stackfit','gt10.87_nage_flxer_spec',
#                                flxerr ,res=12,fsps=True)

# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_nage_noer_stackfit_chidata.fits', np.array(tau),metal,newage)
# onesig,twosig=Likelihood_contours(newage,metal,Pr)
# levels=np.array([twosig,onesig])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# plt.show()
# plt.savefig('../research_plots/flxer_0-6-7_spec_lh.png')
# plt.close()
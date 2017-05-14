from spec_id  import Stack_spec_normwmean, Stack_model_normwmean, Scale_model, Identify_stack,\
    Analyze_Stack_avgage, Likelihood_contours, Analyze_Stack, Model_fit_stack_normwmean,\
    Model_fit_stack_MCerr_bestfit_nwmean
from time import time
import seaborn as sea
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
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Model_fit_stack_normwmean_scaled(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_normwmean(speclist,speczs,wv_range)

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

    pklname='%s.pkl' % pkl_name

    if os.path.isfile(pklname)==False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
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
                S1 = Scale_model(fl, err, mf)
                chigrid[i][ii][iii]=Identify_stack(fl,err,mf*S1)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

def Get_repeats(x,y):
    # z=[x,y]
    # tz=np.transpose(z)
    tz=[[x[U],y[U]] for U in range(len(x))]
    size=np.zeros(len(tz))
    for i in range(len(size)):
        # size[i]=len(np.argwhere(tz==tz[i]))/2
        size[i]=tz.count(tz[i])
        # print len(np.argwhere(tz == tz[i]))

    size/=5.
    return size

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871<=lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#      1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
newage=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
        2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]

tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""model fit"""
M,A=np.meshgrid(metal,newage)
#
# # Model_fit_stack_MCerr_bestfit_nwmean(speclist[IDS],np.array(tau),metal,age,rshift[IDS],np.arange(3250,5500,10),
# #                                      'gt10.87_fsps_newdata_mcerr','pickled_mstacks/gt10.87_fsps_newdata_spec.pkl',
# #                                      repeats=1000)
#
# Model_fit_stack_normwmean(speclist[IDS],tau,metal,newage,rshift[IDS],np.arange(3400,5500,10),
#                          'gt10.87_fsps_10_test_stackfit','ls10.87_fsps_10_test_spec',res=10,fsps=True)

# merr,aerr=np.array(Readfile('gt10.87_fsps_nage_flxer_mcerr.dat',1))
# s=Get_repeats(merr,aerr)
#
# print set(s*5)
# #
Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_10_test_stackfit_chidata.fits', np.array(tau),metal,newage)
onesig,twosig=Likelihood_contours(newage,metal,Pr)
levels=np.array([twosig,onesig])
# levels=np.array([  129.92896389 , 1272.84392063])
print levels
plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.scatter(merr,aerr,s=s,color='k',label='Best fit of 1 $\sigma$\nperturbations to stack')
plt.plot(bfmetal,bfage,'cp',ms=5,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.minorticks_on()
plt.xlabel('Metallicity (Z$_\odot$)')
plt.ylabel('Age (Gyrs)')
plt.legend()
plt.show()

"""stack"""
[flxerr]=np.array(Readfile('flx_err_in2.dat'))
#
flist=[]
for i in range(len(rshift[IDS])):
    flist.append('../../../fsps_models_for_fit/models/m0.0096_a2.68_t0_z%s_model.dat' % rshift[IDS][i])
#
wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3400,5500,12))
print wv
# wv2,fl2,er2=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3394,5500,12))
# mwv,mfl,mer=Stack_model_normwmean(speclist[IDS],flist,rshift[IDS],np.arange(3250,5500,12))
# mwv2,mfl2,mer2=Stack_model_normwmean(speclist[IDS],flist,rshift[IDS],np.arange(3394,5500,12))
#
# ner=np.sqrt(er**2+(fl*flxerr)**2)
#
# plt.plot(mwv[mwv>=3394],mfl[mwv>=3394]*1000,label='<10.87 stack')
# plt.plot(mwv2,mfl2*1000-mfl[mwv>=3394]*1000)
# # plt.plot(wv,ner*1000,label='error')
# plt.errorbar(wv,fl*1000,ner*1000,fmt='o',ms=5,label='>10.87 stack')
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(3250,5500)
# plt.ylabel('Relative Flux',size=20)
# plt.xlabel('Wavelength ($\AA$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.plot(mwv,mfl*1000)
# plt.legend(loc=4,fontsize=15)
# plt.show()
# plt.savefig('../research_plots/gt10.87_bfit_10-12.png')
# plt.close()
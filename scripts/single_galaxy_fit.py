import numpy as np
import sympy as sp
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from astropy.io import fits
import cPickle
import os
from spec_id import Norm_P_stack,Analyze_Stack_avgage,Likelihood_contours,Identify_stack,\
    Analyze_Stack, Scale_model,Model_fit_stack_normwmean,Stack_spec_normwmean,Stack_model_normwmean
from glob import glob
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def NP_total(PR,metal,age):
    a=np.zeros(len(age))
    for i in range(len(age)):
        a[i]=np.trapz(PR[i],metal)
    C=np.trapz(a,age)
    ####### get best fit values
    [idmax] = np.argwhere(PR == np.max(PR))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[0]], metal[idmax[1]])
    return PR/C,age[idmax[0]], metal[idmax[1]]

def P_stack(tau,metal,age,chi):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    prob=[]
    for i in range(len(metal)):
        preprob1=[]
        for ii in range(len(age)):
            preprob2=[]
            for iii in range(len(tau)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii]/2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all tau
    ######## End up with age vs metallicity matricies
    ######## use unlogged tau
    ultau=np.append(0,np.power(10,tau[1:]-9))
    M = []
    for i in range(len(metal)):
        A=[]
        for ii in range(len(age)):
            T=[]
            for iii in range(len(tau) - 1):
                T.append(sp.N((ultau[iii + 1] - ultau[iii]) * (prob[i][ii][iii] + prob[i][ii][iii+1]) / 2))
            A.append(sp.mpmath.fsum(T))
        M.append(A)

    ######## Integrate over metallicity to get age prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(metal)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (M[i][ii] + M[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(metal) - 1):
        preC3.append(sp.N((metal[i + 1] - metal[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = 1

    ######## Create normal prob grid
    P = []
    for i in range(len(metal)):
        preP=[]
        for ii in range(len(age)):
            preP.append(M[i][ii]/C)
        P.append(np.array(preP).astype(np.float128))

    return P

def Analyze_Stack_avgage_non_norm(chifits, tau, metal, age):
    ####### Read in file
    dat = fits.open(chifits)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.transpose(chi)

    scale = Readfile('tau_scale.dat', 1)

    overhead = []
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead.append(sum(amt))

    newchi = []
    for i in range(len(chi)):
        if i == 0:
            iframe = chi[i]
        else:
            iframe = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
        newchi.append(iframe)
    newchi = np.transpose(newchi)

    ####### Create probablity marginalized over tau
    prob = np.array(P_stack(tau, metal, age, newchi)).astype(np.float128)

    for u in prob:
        print u

    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]

def Analyze_Stack_avgage_2(chi, tau, metal, age):

    scale = Readfile('tau_scale.dat', 1)

    overhead = []
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead.append(sum(amt))

    newchi = []
    for i in range(len(chi)):
        if i == 0:
            iframe = chi[i]
        else:
            iframe = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
        newchi.append(iframe)
    newchi = np.transpose(newchi)

    ####### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P_stack(tau, metal, age, newchi)).astype(np.float128)

    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]

def Single_gal_fit(spec, tau, metal, A, specz, name, fsps=False):
    #############Read in spectra#################
    wv,fl,err=np.array(Readfile(spec,1))

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
                    a.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], specz))
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
                    a.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], specz))
                m.append(a)
            modellist.append(m)

    ##############Create chigrid and add to file#################
    chigrid=np.zeros([len(metal),len(A),len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv,mf,merr = np.array(Readfile(modellist[i][ii][iii],1))
                imf=interp1d(mwv,mf)(wv)
                C=Scale_model(fl,err,imf)
                chigrid[i][ii][iii]=Identify_stack(fl,err,imf*C)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)
    ################Write chigrid file###############

    hdulist.writeto(chifile)
    print 'Done!'
    return

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

M,A=np.meshgrid(metal,age)

"""fit all galaxies w/wout error"""
# wv,fl,er=Readfile(speclist[6])
# wv_flx,flx_er=Readfile('flx_err/')
#
#
# plt.plot(wv,fl)
# plt.plot(wv,er)
# plt.show()
flist=[]
for i in range(len(rshift[IDA])):
    flist.append('../../../fsps_models_for_fit/models/m0.0096_a2.64_t0_z%s_model.dat' % rshift[IDA][i])
#
wv,fl,er=Stack_spec_normwmean(speclist[IDA],rshift[IDA],np.arange(3250,5500,10))
wv2,fl2,er2=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5500,10))
mwv,mfl,mer=Stack_model_normwmean(speclist[IDA],flist,rshift[IDA],np.arange(3250,5500,10))

plt.plot(wv,fl)
plt.plot(wv2,fl2)
# plt.plot(mwv,mfl)
plt.show()

"""Add"""
# chigrid = np.zeros([len(metal), len(A), len(tau)])
#
# for i in range(len(speclist[IDS])):
#     Single_gal_fit(speclist[IDS][i],tau,metal,age,rshift[IDS][i],'%s_sg_fit' % ids[IDS][i],fsps=True)
#     print ids[IDS][i]
#     dat = fits.open('chidat/%s_sg_fit_chidata.fits' % ids[IDS][i])
#     chi = []
#     for ii in range(len(metal)):
#         chi.append(dat[ii + 1].data)
#     chi = np.array(chi)
#     chigrid=chigrid+chi
#
# Pr, bfa, bfm = Analyze_Stack_avgage_2(chigrid.T, np.array(tau), metal, age)
# onesig, twosig = Likelihood_contours(age, metal, Pr)
# levels = np.array([twosig, onesig])
# plt.contour(M, A, Pr,levels, colors='k', linewidths=2)
# plt.contourf(M, A, Pr, 40, cmap=cmap)
# plt.plot(bfm,bfa,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfa,np.round(bfm/0.019,2)))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.legend(fontsize=15)
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.xticks([0, .005, .01, .015, .02, .025, .03],
#            np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.019, 2))
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ind_gal_add_LH.png')
# plt.close()

"""Multiply with normalized probabilties"""
# prob=np.ones([len(age),len(metal)])
# for i in range(len(speclist[IDS])):
#     Pr, bfa, bfm= Analyze_Stack_avgage('chidat/%s_sg_fit_chidata.fits' % ids[IDS][i],np.array(tau),metal,age)
#     prob=prob*Pr
#
# nprob,bfage,bfmetal=NP_total(prob,metal,age)
# onesig, twosig = Likelihood_contours(age, metal, nprob)
# levels = np.array([twosig, onesig])
# plt.contour(M, A, nprob,levels, colors='k', linewidths=2)
# plt.contourf(M, A, nprob, 40, cmap=cmap)
# plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Z/Z$_\odot$',size=20)
# plt.ylabel('Average Age (Gyrs)',size=20)
# plt.legend(fontsize=15)
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.xticks([0, .005, .01, .015, .02, .025, .03],
#            np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.019, 2))
# plt.gcf().subplots_adjust(bottom=0.16)
# # plt.show()
# plt.savefig('../research_plots/ind_gal_mult_NPLH.png')
# plt.close()
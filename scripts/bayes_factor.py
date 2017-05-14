from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sympy as sp
from astropy.io import fits
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def B_factor(input_chi_file,tau,metal,age):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    dat = fits.open(input_chi_file)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

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

    C = sp.mpmath.fsum(preC3)

    return C

age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]

metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0190, 0.0240, 0.0300]

metalB = np.array([.0001, .0004, .004, .008, .02])

tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])

ipfsps='chidat/gt10.87_fsps_btest_stackfit_chidata.fits'
ipbc03='chidat/gt10.87_bc03_stackfit_chidata.fits'

b10=float(B_factor(ipfsps,tau,metal,age)/B_factor(ipbc03,tau,metalB,age))

print np.log10(b10)
print 2*np.log(b10)#
#
# ipfsps='chidat/gt10.87_fsps_stackfit_chidata.fits'
# ipbc03='chidat/gt10.87_bc03_2_stackfit_chidata.fits'
#
# b10=float(B_factor(ipfsps,tau,metal,age)/B_factor(ipbc03,tau,[.0001, .0004, .004, .008, .02, .05],age))
#
# print np.log10(b10)
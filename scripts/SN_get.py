import numpy as np
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from spec_id import Stack_spec_normwmean,Model_fit_stack_normwmean,Likelihood_contours,Analyze_Stack_avgage
from astropy.io import ascii
from astropy.table import Table
from glob import glob
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Signal_to_Noise(wave,flux,error):
    S=np.trapz(flux,wave)
    N=np.trapz(error,wave)
    return S/N

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDA=[]  # all masses in sample
IDL=[]  # low mass sample
IDH=[]  # high mass sample

for i in range(len(ids)):
    if 10.0<=lmass[i] and 1<rshift[i]<1.75:
        IDA.append(i)
    if 10.971>lmass[i] and 1<rshift[i]<1.75:
        IDL.append(i)
    if 10.971<lmass[i] and 1<rshift[i]<1.75:
        IDH.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
     2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""make sn list"""
sn=np.zeros(len(IDA))
for i in range(len(IDA)):
    wv,fl,er=Readfile(speclist[IDA][i])
    sn[i]=Signal_to_Noise(wv,fl,er)

snh=np.zeros(len(IDH))
for i in range(len(IDH)):
    wv,fl,er=Readfile(speclist[IDH][i])
    snh[i]=Signal_to_Noise(wv,fl,er)

snl=np.zeros(len(IDL))
for i in range(len(IDL)):
    wv,fl,er=Readfile(speclist[IDL][i])
    snl[i]=Signal_to_Noise(wv,fl,er)

# dat=Table([ids[IDA],speclist[IDA],lmass[IDA],rshift[IDA],sn],
#           names=['ids','specfile','lmass','redshift','S/N'])
# ascii.write(dat,'sn_list.dat')

wvh,flh,erh=Stack_spec_normwmean(speclist[IDH],rshift[IDH],np.arange(3250,5500,10))
wvl,fll,erl=Stack_spec_normwmean(speclist[IDL],rshift[IDL],np.arange(3250,5500,10))

print sum(sn)
print sum(snh)
print sum(snl)
print sum(snh)/sum(snl)

IDxh=[U for U in range(len(wvh)) if 4000<wvh[U]<4200]
IDxl=[U for U in range(len(wvl)) if 4000<wvl[U]<4200]

print np.trapz(flh[IDxh],wvh[IDxh])/np.trapz(erh[IDxh],wvh[IDxh])
print np.trapz(fll[IDxl],wvl[IDxl])/np.trapz(erl[IDxl],wvl[IDxl])
print (np.trapz(flh[IDxh],wvh[IDxh])/np.trapz(erh[IDxh],wvh[IDxh]))/(np.trapz(fll[IDxl],wvl[IDxl])/np.trapz(erl[IDxl],wvl[IDxl]))

# plt.plot(wvh,flh/erh)
# plt.plot(wvl,fll/erl)
plt.plot(wvh,flh)
plt.plot(wvl,fll)
plt.plot(wvh,erh)
plt.plot(wvl,erl)
plt.show()

"""fit S/N"""
# M,A=np.meshgrid(metal,age)
#
# Model_fit_stack_normwmean(speclist[IDH],tau,metal,age,rshift[IDH],np.arange(3250,5250,10),
#                          'gt10.97_fsps_10_stackfit','gt10.97_fsps_10_spec',res=10,fsps=True)
#
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.97_fsps_10_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',ms=5,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/gt10.97_fsps_10_lh.png')
# plt.close()
#
# Model_fit_stack_normwmean(speclist[IDL],tau,metal,age,rshift[IDL],np.arange(3400,5250,10),
#                          'lt10.97_fsps_10_stackfit','lt10.97_fsps_10_spec',res=10,fsps=True)
#
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/lt10.97_fsps_10_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',ms=5,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/lt10.97_fsps_10_lh.png')
# plt.close()

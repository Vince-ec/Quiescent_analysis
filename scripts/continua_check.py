from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from spec_id import Stack_spec
from astropy.io import fits
from scipy.interpolate import interp1d
import seaborn as sea
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Get_flux_nocont(FILE,z):
    w,f,e=np.array(Readfile(FILE,1))
    w/=(1+z)

    m2r = [3175, 3280, 3340, 3515, 3550, 3650, 3710, 3770, 3800, 3850,
            3910, 3989, 3991, 4030, 4082, 4122, 4250, 4385, 4830, 4930, 4990, 5030, 5109, 5250]

    Mask=np.zeros(len(w))
    for i in range(len(Mask)):
        if m2r[0]<=w[i]<=m2r[1]:
            Mask[i]=1
        if m2r[2]<=w[i]<=m2r[3]:
            Mask[i]=1
        if m2r[4]<=w[i]<=m2r[5]:
            Mask[i]=1
        if m2r[6]<=w[i]<=m2r[7]:
            Mask[i]=1
        if m2r[8]<=w[i]<=m2r[9]:
            Mask[i]=1
        if m2r[8]<=w[i]<=m2r[9]:
            Mask[i]=1
        if m2r[10]< w[i]<=m2r[11]:
            Mask[i] = 1
        if m2r[12]<=w[i]<=m2r[13]:
            Mask[i]=1
        if m2r[14]<=w[i]<=m2r[15]:
            Mask[i]=1
        if m2r[16]<=w[i]<=m2r[17]:
            Mask[i]=1
        if m2r[18] <= w[i] <= m2r[19]:
            Mask[i] = 1
        if m2r[20] <= w[i] <= m2r[21]:
            Mask[i] = 1
        if m2r[22] <= w[i] <= m2r[23]:
            Mask[i] = 1


    maskw = np.ma.masked_array(w, Mask)

    x3, x2, x1, x0 = np.ma.polyfit(maskw, f, 3, w=1/e**2)
    C0 = x3 * w ** 3 + x2 * w ** 2 + x1 * w + x0

    f/=C0
    e/=C0

    return w, f, e

def Stack_spec2(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave,flux,error=np.array(Get_flux_nocont2(spec[i],redshifts[i]))
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        flgrid[i][mask]=ifl(wv[mask])
        errgrid[i][mask]=ier(wv[mask])
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

# for i in range(len(speclist)):
#     wv,fl,er=np.array(Readfile(speclist[i],1))
#     wv/=(1+rshift[IDS][i])
#     plt.plot(wv,fl)
#     plt.plot(wv,er)
#     plt.title(speclist[i])
#     plt.axvspan(3550, 3650,alpha=.2)
#     plt.axvspan(3710, 3770,alpha=.2)
#     plt.axvspan(3800, 3850,alpha=.2)
#     plt.axvspan(3910, 4030,alpha=.2)
#     plt.axvspan(4082, 4122,alpha=.2)
#     plt.axvspan(4250, 4400,alpha=.2)
#     plt.axvspan(4830, 4930,alpha=.2)
#     plt.axvspan(4990, 5030,alpha=.2)
#     plt.axvspan(5110, 5250,alpha=.2)
#     plt.axvspan(5530, 5650,alpha=.2)
#     plt.show()

wv,fl,er=Stack_spec(speclist,rshift[IDS],np.arange(3250,5500,5))
wv2,fl2,er2=Stack_spec2(speclist,rshift[IDS],np.arange(3250,5500,5))

mwv1,mfl1,mer1=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.0_model.dat',1))
mwv2,mfl2,mer2=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.1_model.dat',1))
mwv3,mfl3,mer3=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.2_model.dat',1))
mwv4,mfl4,mer4=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.3_model.dat',1))
mwv5,mfl5,mer5=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.4_model.dat',1))
mwv6,mfl6,mer6=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.5_model.dat',1))
mwv7,mfl7,mer7=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.6_model.dat',1))
mwv8,mfl8,mer8=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.7_model.dat',1))
mwv9,mfl9,mer9=np.array(Readfile('../../../fsps_models_for_fit/models/m0.019_a2.11_t8.0_z1.8_model.dat',1))

ID1=[U for U in range(len(mwv1)) if 7900<mwv1[U]<11300]

plt.plot(mwv1[ID1]/2,mfl1[ID1])
plt.plot(mwv2[ID1]/2.1,mfl2[ID1])
plt.plot(mwv3[ID1]/2.2,mfl3[ID1])
plt.plot(mwv4[ID1]/2.3,mfl4[ID1])
plt.plot(mwv5[ID1]/2.4,mfl5[ID1])
plt.plot(mwv6[ID1]/2.5,mfl6[ID1])
plt.plot(mwv7[ID1]/2.6,mfl7[ID1])
plt.plot(mwv8[ID1]/2.7,mfl8[ID1])
plt.plot(mwv9[ID1]/2.8,mfl9[ID1])

plt.axvline(2910)
plt.axvline(5650)
plt.axvspan(3175, 3280, alpha=.2)
plt.axvspan(3340, 3515, alpha=.2)
plt.axvspan(3550, 3650, alpha=.2)
plt.axvspan(3710, 3770, alpha=.2)
plt.axvspan(3800, 3850, alpha=.2)
plt.axvspan(3910, 3989, alpha=.2)
plt.axvspan(3991, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4385, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(2900,6000)
plt.show()

plt.plot(wv,fl)
plt.plot(wv2,fl2)
plt.plot(wv2,np.ones(len(wv)),'k--',alpha=.2)
plt.axvspan(3175, 3280, alpha=.2)
plt.axvspan(3340, 3515, alpha=.2)
plt.axvspan(3550, 3650, alpha=.2)
plt.axvspan(3710, 3770, alpha=.2)
plt.axvspan(3800, 3850, alpha=.2)
plt.axvspan(3910, 3989, alpha=.2)
plt.axvspan(3991, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4385, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.show()

# plt.plot(wv,1/er**2)
# plt.plot(wv2,1/er2**2)
# plt.plot(wv3,1/er3**2)
# plt.show()


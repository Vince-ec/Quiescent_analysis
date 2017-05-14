from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.interpolate import interp1d
from spec_id import Stack_spec_normwmean,Stack_model_normwmean
import cPickle
import seaborn as sea
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in" ,"ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Binchi(wv,chi):
    bins=[[3250, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
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

bins = [[3250, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
        [4400, 4830], [4830, 4930], [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250],[5250,5500]]

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

zlist= []
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput=int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

"""spec stacking"""
wv, fl, err = Stack_spec_normwmean(speclist, rshift[IDS], np.arange(3400,5250,10))

modmetal=[0.0068, 0.0077, 0.0085, 0.0096, 0.0106, 0.012, 0.0132, 0.014, 0.0150, 0.0164, 0.018]
modage=[2.11, 2.11, 2.11, 2.11, 2.11, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42]

mfl=np.zeros([len(modmetal),len(wv)])

for i in range(len(modmetal)):
    flist = []
    for ii in range(len(zlist)):
        flist.append('../../../fsps_models_for_fit/models/m%s_a%s_t0_z%s_model.dat' % (modmetal[i],modage[i],zlist[ii]))
    fwv,mfl[i],fe=Stack_model_normwmean(speclist,flist, rshift[IDS], zlist, np.arange(wv[0],wv[-1]+10,10))

"""reg Test"""
red=np.zeros(len(bins))
for i in range(len(bins)):
    r=[]
    for ii in range(len(wv)):
        if bins[i][0]<=wv[ii]<bins[i][1]:
            r.append(ii)
    red[i]=len(r)

sr=np.zeros(len(metal))
bwv=[]
bchi=[]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)

for i in range(len(modmetal)):
    srwv=np.abs((fl - mfl[i]) / err)
    sr[i]=sum(srwv)
    bw,bc=Binchi(wv,srwv)
    bwv.append(bw)
    bchi.append(bc)
    ax1.plot(bwv[i],bchi[i]/red[i],'o',color='#226666', alpha=float(i)/len(modmetal),ms=6,zorder=1)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
ax1.set_ylabel('$\chi ^2$',size=15)
ax2.set_ylabel('Relative Flux',size=15)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax2.tick_params(axis='both', which='major', labelsize=13)
plt.minorticks_on()
plt.gcf().subplots_adjust(bottom=0.16)
# ax1.legend(loc=3,fontsize=13)
plt.show()
# plt.savefig('../research_plots/mcomp_chifeat_nwm.png')
# plt.close()

# featstd=[np.std(np.transpose(bchi)[U]) for U in range(len(np.transpose(bchi)))]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()
# ax2.plot(wv,fl,'k',lw=1,alpha=.2,zorder=0)
# ax1.plot(bwv[0],featstd,'o',ms=6)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# ax1.set_xlabel('Restframe Wavelength $\AA$',size=15)
# ax1.set_ylabel('$\sigma / \lambda$',size=15)
# ax2.set_ylabel('Relative Flux',size=15)
# ax1.tick_params(axis='both', which='major', labelsize=13)
# ax2.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# plt.savefig('../research_plots/mcomp_std_nwm.png')
# plt.close()

from spec_id  import Stack_spec,Stack_model
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_zps2.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]

for i in range(len(zsmax)):
    zlist.append(int(zsmax[i] * 100) / 5 / 20.)
for i in range(len(bins)):
    b = []
    for ii in range(len(zlist)):
        if bins[i] == zlist[ii]:
            b.append(ii)
    if len(b) > 0:
        zcount.append(len(b))
zbin = sorted(set(zlist))

##########################
blist=[]
for i in range(len(zbin)):
    blist.append('../../../bc03_models_for_fit/models/m0.008_a1.62_z%s_model.dat' % zbin[i])

flist=[]
for i in range(len(zbin)):
    if zbin[i]==1:
        zbin[i]=int(1)
    flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_z%s_model.dat' % zbin[i])

# wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))

# bwv,bs,be=Stack_model(blist,zbin,zcount,np.arange(3250,5550,5))
# fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
#
# plt.plot(wv,s)
# plt.plot(bwv,bs)
# plt.plot(fwv,fs)
# plt.show()


dat1=np.array(Readfile('chidat/stack_fit_fsps_nored_chidata.dat',1),dtype=np.float128)
dat2=np.array(Readfile('chidat/stackfit_bc03_chidata.dat',1),dtype=np.float128)

age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]

metal1 = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
metal2 = np.array([.0001, .0004, .004, .008, .02, .05])
P1=np.exp(-dat1/2)
P2=np.exp(-dat2/2)
M1,A=np.meshgrid(metal1,age)
M2,A=np.meshgrid(metal2,age)

a1=np.zeros(len(age))
a2=np.zeros(len(age))
for i in range(len(age)):
    a1[i]=np.trapz(P1[i],metal1)
    a2[i]=np.trapz(P2[i],metal2)
C1=np.trapz(a1,age)
C2=np.trapz(a2,age)
a1/=C1
a2/=C2

Pt1=np.transpose(P1)
Pt2=np.transpose(P2)
m1=np.zeros(len(metal1))
m2=np.zeros(len(metal2))
for i in range(len(metal1)):
    m1[i]=np.trapz(Pt1[i],age)
for i in range(len(metal2)):
    m2[i]=np.trapz(Pt2[i],age)
m1/=C1
m2/=C2

a1ex=np.trapz(age*a1,age)
a2ex=np.trapz(age*a2,age)
m1ex=np.trapz(metal1*m1,metal1)
m2ex=np.trapz(metal2*m2,metal2)

mticks=np.array([0,.01,.02,.03,.04,.05])

plt.plot(age,a1,label='FSPS')
plt.axvline(a1ex,linestyle='-.',color=sea.color_palette('muted')[2],label='$\langle t \\rangle _{FSPS}$')
plt.plot(age,a2,label='BC03')
plt.axvline(a2ex,linestyle='--',color=sea.color_palette('muted')[2],label='$\langle t \\rangle _{BC03}$')
plt.xlabel('Age (Gyrs)',size=13)
plt.ylabel('P(t)',size=13)
# plt.plot(metal1,m1,label='FSPS')
# plt.axvline(m1ex,linestyle='-.',color=sea.color_palette('muted')[2],label='$\langle Z \\rangle _{FSPS}$')
# plt.plot(metal2,m2,label='BC03')
# plt.axvline(m2ex,linestyle='--',color=sea.color_palette('muted')[2],label='$\langle Z \\rangle _{BC03}$')
# plt.xlim(0,.05)
# plt.xticks(mticks,np.round(mticks/0.02,2))
# plt.xlabel('Z/Z$_\odot$',size=13)
# plt.ylabel('P(Z)',size=13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=13)
plt.show()
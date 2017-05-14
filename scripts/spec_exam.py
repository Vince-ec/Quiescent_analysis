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

# rmlist1=[]
# for i in range(len(zbin)):
#     rmlist1.append('../../../fsps_models_for_fit/models/m0.012_a4.62_z%s_model.dat' % zbin[i])
#
# rmlist2=[]
# for i in range(len(zbin)):
#     rmlist2.append('../../../fsps_models_for_fit/models/m0.019_a0.84_z%s_model.dat' % zbin[i])
#
# mlist1=[]
# for i in range(len(zbin)):
#     mlist1.append('../../../fsps_models_for_fit/models/m0.012_a1.77_z%s_model.dat' % zbin[i])
#
# mlist2=[]
# for i in range(len(zbin)):
#     mlist2.append('../../../fsps_models_for_fit/models/m0.015_a1.47_z%s_model.dat' % zbin[i])

bfmlist=[]
for i in range(len(zbin)):
    bfmlist.append('../../../bc03_models_for_fit/models/m0.004_a4.05_z%s_model.dat' % zbin[i])

wv,s,e=Stack_spec(speclist,zsmax,np.arange(3250,5550,5))
# rwv1,rs1,re1=Stack_model(rmlist1,zbin,zcount,np.arange(3250,5550,5))
# rwv2,rs2,re2=Stack_model(rmlist2,zbin,zcount,np.arange(3250,5550,5))
# wv1,s1,e1=Stack_model(mlist1,zbin,zcount,np.arange(3250,5550,5))
# wv2,s2,e2=Stack_model(mlist2,zbin,zcount,np.arange(3250,5550,5))
bwv,bs,be=Stack_model(bfmlist,zbin,zcount,np.arange(3250,5550,5))

inrwv1,inrwv2,inwv1,inwv2,inbwv=np.zeros([5,len(wv)])

for i in range(len(wv)):
    # if s[i]-e[i]<=rs1[i]<=s[i]+e[i]:
    #     inrwv1[i]=1
    # if s[i]-e[i]<=rs2[i]<=s[i]+e[i]:
    #     inrwv2[i] = 1
    # if s[i]-e[i]<=s1[i]<=s[i]+e[i]:
    #     inwv1[i] = 1
    # if s[i]-e[i]<=s2[i]<=s[i]+e[i]:
    #     inwv2[i] = 1
    if s[i]-e[i]<=bs[i]<=s[i]+e[i]:
        inbwv[i] = 1

# print sum(inrwv1)/float(len(wv))
# print sum(inrwv2)/float(len(wv))
# print sum(inwv1)/float(len(wv))
# print sum(inwv2)/float(len(wv))
print sum(inbwv)/float(len(wv))



##########################
##########################
plt.fill_between(wv,s-e,s+e,color=sea.color_palette('muted')[5],alpha=.9)
plt.plot(wv,s)
plt.plot(bwv,bs)
# plt.plot(rwv1,rs1)
# plt.plot(rwv2,rs2)
# plt.plot(wv1,s1)
# plt.plot(wv2,s2)
plt.title('Best Fit', size=13)
plt.xlabel('$\lambda$',size=13)
plt.ylabel('Flux',size=13)
plt.xlim(3250,5550)
# plt.ylim(0.4, 1.5)
plt.show()
# plt.savefig('plots/stack_comp_bfit.png')
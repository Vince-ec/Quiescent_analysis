from spec_id  import Stack_spec, Stack_model, Model_fit_stack, Analyze_Stack,Likelihood_contours
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)
nlist=glob('spec_stacks/*')

IDS=[]

# print len(ids)

for i in range(len(ids)):
    if 10.87<lmass[i]:
        IDS.append(i)

# print len(IDS)
print rshift[IDS]
# print ids[IDS]
# print nlist[IDS]

# plt.plot(lmass[IDS],rshift[IDS],'o')
# plt.xlabel('lmass')
# plt.ylabel('redshift')
# plt.show()

speclist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i]==nlist[ii][12:18]:
            speclist.append(nlist[ii])
# print speclist
#
#
# metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
#          0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]
metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0190, 0.0240, 0.0300]
# metal = np.array([.0001, .0004, .004, .008, .02, .05])
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]
# #
M,A=np.meshgrid(metal,age)
#
speczs = np.round(rshift[IDS], 2)
#
#
Model_fit_stack(speclist,tau,metal,age,speczs,np.arange(3800,5200,5),
                'gt10.87_fsps_btest_stackfit','gt10.87_fsps_btest_spec',fsps=True)
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1, as_cmap=True)
Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_btest_stackfit_chidata.fits', np.array(tau),metal,age)
onesig,twosig=Likelihood_contours(age,metal,Pr)
levels=np.array([twosig,onesig])
# levels=np.array([  5.68298695,  26.92653341])
print levels
plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
plt.contourf(M,A,Pr,40,cmap=colmap)
plt.plot(bfmetal,bfage,'cp')
plt.show()
# plt.close()
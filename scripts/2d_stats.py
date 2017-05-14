from astropy.io import fits
import numpy as np
from spec_id import P
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

dat=fits.open('chidat/s39804_specid_chidata.fits')
age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# metal = np.array([.0001, .0004, .004, .008, .02])
tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])

chi = []
for i in range(len(metal)):
    chi.append(dat[i + 1].data)
chi = np.array(chi)

prob = P(chi)

chigr=[]
for i in range(len(metal)):
    acomp=[]
    for ii in range(len(age)):
        acomp.append(np.trapz(prob[i][ii],np.power(10,tau-9)))
    chigr.append(acomp)
prob=chigr

M,A=np.meshgrid(metal,age)

###########################

m=np.zeros(len(metal))

for i in range(len(metal)):
    m[i]=np.trapz(prob[i],age)
C0=np.trapz(m,metal)
m/=C0
prob/=C0

probt=np.transpose(prob)

a2=np.linspace(min(age),max(age),100)
m2=np.linspace(min(metal),max(metal),100)

iprob=interp2d(metal,age,probt)
P2=iprob(m2,a2)

a=np.zeros(len(a2))

for i in range(len(P2)):
    a[i]=np.trapz(P2[i],m2)
C1=np.trapz(a,a2)
P2/=C1

M2,A2=np.meshgrid(m2,a2)

pbin=np.linspace(0,np.max(P2),5000)
pbin=pbin[::-1]

for i in range(len(pbin)):
    p=np.array(P2,dtype=np.float128)
    p[p<=pbin[i]]=0
    a = np.zeros(len(a2))

    for ii in range(len(p)):
        a[ii] = np.trapz(p[ii], m2)
    C0 = np.trapz(a, a2)
    if .675<=C0<=.685:
        print C0,pbin[i]
    if .948 <= C0 <= .952:
        print C0, pbin[i]



import numpy as np
from vtl.Readfile import Readfile
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

z=np.array([])
zz=0.01
while zz<6:
    z=np.append(z,zz)
    zz=zz+.01*(1+zz)

zp=np.linspace(0.9,2.1,121)
mu=[1.22,1.27,1.18,1.36,1.16,1.65,1.215,1.222,1.49,1.7,1.63,1.65,1.018,1.67,1.08,1.227,1.043,1.042,1.12,1.14]
sig=[.03,.03,.03,.04,.03,.05,.03,.03,.09,.09,.04,.05,.04,.04,.03,0.04,.02,.03,.03,.03]

fn=glob('Pofz/*')

for i in range(len(fn)):
    [pz]=np.array(Readfile(fn[i],1))
    truepz=np.exp(-pz/2)
    c0=np.trapz(truepz,z)
    truepz/=c0
    photodist=1/(sig[i]*np.sqrt(2*np.pi))*np.exp(-((zp-mu[i])**2)/(2*sig[i]**2))
    tname=fn[i][5:10]
    plt.plot(z,truepz,label='Eazy P(z)')
    plt.plot(zp,photodist,label='Gaussian P(z)')
    plt.xlim(.9,2.1)
    plt.xlabel('z')
    plt.ylabel('P(z)')
    plt.legend()
    plt.title(tname)
    plt.savefig('Pofz/%s_pzcomp.png' %tname)
    plt.close()
    # plt.show()

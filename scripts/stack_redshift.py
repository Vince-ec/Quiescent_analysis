from spec_id import Scale_model
from glob import glob
from scipy.interpolate import interp1d
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from spec_id import Error
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

def Identify(Datafile, Models):
    dwv, dfl, derr= np.array(Readfile(Datafile,1))
    chi = np.zeros(len(Models))

    for i in range(len(Models)):
        mwv1, mfl1, mer = np.array(Readfile(Models[i],1))
        mfl = np.array(interp1d(mwv1, mfl1)(dwv))
        C = Scale_model(dfl, derr, mfl)
        x = ((dfl - C * mfl) / derr) ** 2
        chi[i]=np.sum(x)

    return chi

def Redshifts(chi,z, pzfn):
    chi=np.array(chi,dtype=np.float128)
    ###P1###
    p1=np.exp(-chi/2)
    c0=np.trapz(p1,z)
    p1/=c0
    ###P2###
    Pz,red=np.array(Readfile(pzfn,1))
    Pz=np.exp(-Pz/2,dtype=np.float128)
    c1=np.trapz(Pz,red)
    Pz/=c1
    pzinterp=interp1d(red,Pz)
    pzinrange=pzinterp(z)
    ###combined P###
    p2=pzinrange*p1
    c1=np.trapz(p2,z)
    p2/=c1
    ###get red shifts###
    zsmax=z[np.argmin(chi)]
    zschi=chi[np.argmin(chi)]
    zs=np.trapz(z*p1,z)
    zsl,zsh=Error(p1,z)
    zpsmax = z[np.argmax(p2)]
    zpschi=chi[np.argmax(p2)]
    zps = np.trapz(z * p2, z)
    zpsl, zpsh = Error(p2, z)

    # plt.plot(z, p1)
    # plt.plot(z,pzinrange)
    # plt.plot(z,p2)
    # plt.plot(red,Pz)
    # # plt.plot(red,Pz1)
    # plt.axvline(zsmax,color=sea.color_palette('Set2')[3],label='zsmax')
    # plt.axvline(zs,color=sea.color_palette('Set2')[4],label='zps')
    # plt.axvline(zpsmax,color=sea.color_palette('Set2')[5],label='zpsmax')
    # plt.axvline(zps,color=sea.color_palette('Set2')[0],label='zps')
    # plt.xlim(1, 2)
    # plt.legend()
    # plt.show()
    return np.round([zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh],4)

###Create redshift array###
z=np.arange(.9,2.105,.001)
###Get list of models to test against###
modeldat=[]
filepath = '../../../fsps_models_for_fit/models/'
for i in range(len(z)):
    modeldat.append(filepath + 'm0.015_a1.5_t8.0_z%s_model.dat' % z[i])
###Get spectras###
fn=glob('spec_stacks_nov29/*stack.dat')
fnpz=glob('Pofz/*.dat')

for i in range(len(fn)):
    print fn[i]
    print fnpz[i]

##########Create redshift arrays##########
zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.zeros([10,len(fn)])

##########Get redshifts#######
for i in range(len(fn)):
    chi=Identify(fn[i],modeldat)
    zsmax[i], zschi[i], zs[i], zsl[i], zsh[i], zpsmax[i], zpschi[i], zps[i], zpsl[i], zpsh[i]=Redshifts(chi,z,fnpz[i])

##########Write file#########
dat=Table([fn,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh],
          names=['File','zspec_max','zspec_chi','zspec','zspec_el','zspec_eh',
                 'zpspec_max','zpspec_chi','zpspec','zpspec_el','zpspec_eh'])
ascii.write(dat,'stack_redshifts_10-6.dat')

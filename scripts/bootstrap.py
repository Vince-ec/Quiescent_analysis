from spec_id  import Get_flux_nocont
from scipy.interpolate import interp1d
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

def Stack_spec(spec,redshifts, wv_range):

    Data=[]

    for i in range(len(spec)):
        Data.append(np.array(Get_flux_nocont(spec[i],redshifts[i])))

    wv=np.array(wv_range)

    interpdata=[]
    interpweight=[]
    stack=np.zeros(len(wv))
    err=np.zeros(len(wv))

    for i in range(len(Data)):
        wt=(Data[i][2])**(-2)
        interpdata.append(interp1d(Data[i][0],Data[i][1]))
        interpweight.append(interp1d(Data[i][0],wt))

    for i in range(len(wv)):
        flu=np.zeros(len(interpdata))
        wei=np.zeros(len(interpdata))
        for ii in range(len(interpdata)):
            if Data[ii][0][0]<=wv[i]<=Data[ii][0][-1]:
                flu[ii]=interpdata[ii](wv[i])
                wei[ii]=interpweight[ii](wv[i])
        stack[i] = np.sum(wei*flu)/np.sum(wei)
        err[i]=1/np.sqrt(np.sum(wei))

    return wv, stack, err

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

speclist=np.append(speclist[0:18],speclist[19])
zps=np.append(zps[0:18],zps[19])

bs=np.zeros([25,len(np.arange(3700,5500,5))])

for i in range(25):
    IDX=np.random.choice(len(speclist),len(speclist))
    slist=speclist[IDX]
    z=zps[IDX]
    wv,s,e=Stack_spec(slist,z,np.arange(3700,5500,5))
    bs[i]=s

err=np.transpose(bs)

fl=np.array([])
er=np.array([])
for i in range(len(err)):
    fl=np.append(fl,np.mean(err[i]))
    er=np.append(er,np.std(err[i]))

wv,s,e=Stack_spec(speclist,zps,np.arange(3700,5500,5))

plt.fill_between(wv, s - er, s + er, color=sea.color_palette('muted')[5], alpha=.9)
plt.plot(wv,s,lw=1)
plt.xlabel('$\lambda$',size=13)
plt.ylabel('Flux',size=13)
plt.title('zps logmass>10.5',size=13)
plt.xlim(3700,5500)
plt.show()
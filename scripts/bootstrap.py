from spec_id import Galaxy_set
from scipy.interpolate import interp1d
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})



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
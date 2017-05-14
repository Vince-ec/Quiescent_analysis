from spec_id  import Get_flux
from vtl.Readfile import Readfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from spec_id import Model_fit_stack
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

mr = [[3910, 3990], [4082, 4122], [4250, 4330], [4830, 4890], [4990, 5030]]

fn,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)
# z=np.array([1.068,1.13,1.599,1.043,1.04,1.173,1.22,1.958,1.195,1.377,1.152,1.222,1.625,1.678,1.261,1.652,
#             1.215,1.599,1.787,1.027,1.678])

for i in range(len(fn)):

    name=fn[i][12:17]

    wv,fl,er=np.array(Readfile(fn[i],1))
    wv/=(1+zps[i])

    Mask=np.zeros(len(wv))

    for ii in range(len(mr)):
        for iii in range(len(wv)):
            if mr[ii][0]<=wv[iii]<=mr[ii][1]:
                Mask[iii]=1

    maskwv=np.ma.masked_array(wv,Mask)

    x3, x2, x1, x0 = np.ma.polyfit(maskwv, fl, 3,w=1/er**2)
    C0 = x3 * wv ** 3 + x2 * wv ** 2 + x1 * wv + x0

    wv1, fl1, er1 = np.array(Readfile(fn[i], 1))
    wv1 /= (1 + zsmax[i])

    Mask = np.zeros(len(wv1))

    for ii in range(len(mr)):
        for iii in range(len(wv1)):
            if mr[ii][0] <= wv1[iii] <= mr[ii][1]:
                Mask[iii] = 1

    maskwv = np.ma.masked_array(wv1, Mask)

    x3, x2, x1, x0 = np.ma.polyfit(maskwv, fl1, 3, w=1 / er1 ** 2)
    C1 = x3 * wv1 ** 3 + x2 * wv1 ** 2 + x1 * wv1 + x0

    plt.plot(wv,fl)
    plt.plot(wv,C0)
    # plt.plot(wv1, fl1)
    # plt.plot(wv1, C1)
    plt.axvspan(mr[0][0], mr[0][1],color=sea.color_palette('muted')[5],alpha=.9)
    plt.axvspan(mr[1][0], mr[1][1],color=sea.color_palette('muted')[5],alpha=.9)
    plt.axvspan(mr[2][0], mr[2][1],color=sea.color_palette('muted')[5],alpha=.9)
    plt.axvspan(mr[3][0], mr[3][1],color=sea.color_palette('muted')[5],alpha=.9)
    plt.axvspan(mr[4][0], mr[4][1],color=sea.color_palette('muted')[5],alpha=.9)
    plt.xlabel('$\lambda$', size=15)
    plt.ylabel('F$_\lambda$', size=15)
    plt.title(name, size=15)
    plt.savefig('masking_plots/%s.png' % name)
    plt.close()
    # plt.show()
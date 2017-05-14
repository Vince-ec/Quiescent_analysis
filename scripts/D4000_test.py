import vtl.Constants as C
import numpy as np
from scipy.interpolate import interp1d
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

def D4000(wv, flam):
    c=C.c
    h = C.h
    atocm = C.angtocm

    wave = np.multiply(wv, atocm)
    nu = np.divide(c, wave)
    fnu = np.multiply(np.divide(c, np.square(nu)), flam)
    Fnu = interp1d(nu, fnu)

    Dblue = np.divide(c,np.multiply([3850, 3950],atocm))
    Dred = np.divide(c,np.multiply([4000, 4100],atocm))

    dbluenu=np.linspace(Dblue[0],Dblue[1],100)
    drednu=np.linspace(Dred[0],Dred[1],100)

    energy = np.divide(1 / h, dbluenu)
    top1 = np.multiply(Fnu(dbluenu), energy)
    top = np.trapz(top1, dbluenu)
    bottom = np.trapz(energy, dbluenu)
    blue = top / bottom

    energy = np.divide(1 / h, drednu)
    top1 = np.multiply(Fnu(drednu), energy)
    top = np.trapz(top1, drednu)
    bottom = np.trapz(energy, drednu)
    red = top / bottom

    D4=red / blue

    return D4


age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]

modeldat=[]
filepath = '../../../fsps_models_for_fit/models/'
for i in range(len(age)):
    modeldat.append(filepath + 'm0.019_a%s_z1.2_model.dat' % age[i])


d4=np.zeros(len(modeldat))
d4nc=np.zeros(len(modeldat))

for i in range(len(modeldat)):
    w,f,e=np.array(Readfile(modeldat[i],1))
    w/=2.2
    d4[i]=D4000(w,f)
    # plt.plot(w, f)
    mr = [[3910, 3990], [4082, 4122], [4250, 4330], [4830, 4890], [4990, 5030]]

    Mask = np.zeros(len(w))

    for ii in range(len(mr)):
        for iii in range(len(w)):
            if mr[ii][0] <= w[iii] <= mr[ii][1]:
                Mask[iii] = 1

    maskw = np.ma.masked_array(w, Mask)
    x3, x2, x1, x0 = np.ma.polyfit(maskw, f, 3, w=1 / e ** 2)
    C0 = x3 * w ** 3 + x2 * w ** 2 + x1 * w + x0
    f /= C0

    d4nc[i]=D4000(w,f)
    plt.plot(w,f)
plt.xlim(3500,5000)
plt.ylim(0,2)

# plt.plot(d4,d4nc)
# plt.plot([1.2,2],[1.2,2])
# plt.show()

# plt.plot(age,d4)
# plt.plot(age,d4nc)
# plt.show()

# plt.plot(w,f)
# plt.xlim(3500,5000)
# plt.ylim(0,2)
plt.show()
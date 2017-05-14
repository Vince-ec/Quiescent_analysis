from vtl.Readfile import Readfile
from spec_id import Error,P, Likelihood_contours
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sea
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

dat=fits.open('chidat/lt10.93_jan15_fit_chidata.fits')


metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,      ##new age
        2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]                         ##new age
tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10])

"""1D"""
# age=[0.5, 0.84, 1.37, 1.52, 1.72, 1.8, 1.85, 1.95 ,2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.4,2.44,2.48,2.52, 2.56,2.6, 2.64, 2.68,
#      2.7, 2.75, 2.79, 2.81,2.87, 2.95,3.02, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
# chi = []
# for i in range(len(metal)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
#
# prob = P(chi)
# print len(prob),len(prob[0]),len(prob[0][0])
# # print np.argwhere(prob==np.max(prob))
# print np.min(chi)
# x=np.argwhere(prob==np.max(prob))
# print metal[x[0][0]],age[x[0][1]],tau[x[0][2]]
# print chi[x[0][0]][x[0][1]][x[0][2]]

# prob=np.transpose(prob)
# M,A=np.meshgrid(metal,age)

# for U in chi.T[1][::-1]:
#     print np.round(U,1)

# print chi[0]

    # plt.contour(M, A, chi.T[i],colors='k', linewidths=2)
    # plt.contourf(M, A, chi.T[i], 40, cmap=cmap)
    # plt.tick_params(axis='both', which='major', labelsize=17)
    # plt.gcf().subplots_adjust(bottom=0.16)
    # plt.minorticks_on()
    # plt.xlabel('Metallicity (Z$_\odot$)')
    # plt.ylabel('Age (Gyrs)')
    # plt.legend()
    # plt.title('%s' % tau[i] )
    # plt.xticks([0, .005, .01, .015, .02, .025, .03],
    #            np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.019, 2))
    # plt.show()

# for i in range(len(prob)):
#     m = []
#     for ii in range(len(age)):
#         m.append(np.trapz(prob[i][ii],metal))
#     C0=np.trapz(m,age)
#     onesig, twosig = Likelihood_contours(age, metal, prob[i]/C0)
#     levels = np.array([twosig, onesig])
#     # levels=np.array([108.40725535  1212.2568081])
#     print levels
#     plt.contour(M, A, prob[i], levels,colors='k', linewidths=2)
#     plt.contourf(M, A, prob[i], 40, cmap=cmap)
#     plt.tick_params(axis='both', which='major', labelsize=17)
#     plt.gcf().subplots_adjust(bottom=0.16)
#     plt.minorticks_on()
#     plt.xlabel('Metallicity (Z$_\odot$)')
#     plt.ylabel('Age (Gyrs)')
#     plt.legend()
#     plt.title('%s' % tau[i] )
#     plt.xticks([0, .005, .01, .015, .02, .025, .03],
#                np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.019, 2))
#     plt.show()
#     plt.savefig('../research_plots/%s_tau_exam.png' % tau[i])
#     plt.close()

# t=np.zeros(len(tau))
# for i in range(len(tau)):
#     a=np.zeros(len(age))
#     for ii in range(len(age)):
#         a[ii]=np.trapz(prob[i][ii],metal)
#     t[i]=np.trapz(a,age)
# c0=np.trapz(t,np.power(10,tau-9))
# t/=c0
# ext=np.trapz(t*np.power(10,tau-9),np.power(10,tau-9))
#
# print ext
#
# plt.plot(np.power(10,tau),t)
# plt.show()

"""2D"""
# chi = []
# for i in range(len(tau)):
#     chi.append(dat[i + 1].data)
# chi = np.array(chi)
#
# print len(chi),len(chi[0]),len(chi[0][0])
#
# prob = P(chi)
# print len(prob),len(prob[0]),len(prob[0][0])
# print np.argwhere(prob==np.max(prob))
# x=np.argwhere(prob==np.max(prob))
# print tau[x[0][0]],metal[x[0][1]],age[x[0][2]]
# # prob=np.transpose(prob)
# M,A=np.meshgrid(metal,age)
#
# for i in range(len(prob)):
#     plt.contourf(M,A,chi[i].T3,100,cmap='ocean')
#     plt.title('%s' % tau[i] )
#     plt.colorbar()
#     plt.show()
# #
# t=np.zeros(len(tau))
# for i in range(len(tau)):
#     a=np.zeros(len(age))
#     for ii in range(len(age)):
#         a[ii]=np.trapz(prob[i][ii],metal)
#     t[i]=np.trapz(a,age)
# c0=np.trapz(t,np.power(10,tau-9))
# t/=c0
# ext=np.trapz(t*np.power(10,tau-9),np.power(10,tau-9))
#
# print ext
#
# plt.plot(np.power(10,tau),t)
# plt.show()

"""age scaling"""
def Average_age(age,tau):
    t=np.append([0],np.array(age))
    if tau==0:
        sfh=np.ones(len(t))
    else:
        sfh = np.exp(-t / np.power(10, tau - 9))
    avgage=np.zeros(len(age))

    bottom=np.trapz(sfh,t)
    for i in range(len(age)):
        top=np.trapz(t[0:i+2]*sfh[0:i+2],t[0:i+2])
        # print t[0:i+2]
        # avgage[i]=top/bottom
        avgage[i]=top
    return avgage

for i in range(len(tau)):
    Avage=Average_age(age,tau[i])
    print Avage
    plt.plot(np.repeat(i,len(age)),Avage,'o')
plt.show()
# #
plt.plot(age,age*np.exp(-np.array(age)/np.power(10, tau[2] - 9)))
plt.show()

print np.power(10, tau - 9)

age=np.arange(.5,6.1,.1)

print len(age)

a=np.linspace(0,14,100)
aa=np.append([0],age)
# for i in range(len(tau)):
#     scale=np.trapz(np.exp(-a / np.power(10, tau[i] - 9)),a)
#     # print np.trapz(aa*np.exp(-aa / np.power(10, tau[i] - 9)),aa)
#     plt.plot(a,np.exp(-a / np.power(10, tau[i] - 9))/scale,'r',alpha=(float(i)+1)/(len(tau)))
#     t=np.exp(-a / np.power(10, tau[i] - 9))*(a+np.power(10, tau[i] - 9))-np.power(10, tau[i] - 9)
#     b=np.exp(-a / np.power(10, tau[i] - 9))-1
#     print np.power(10, tau[i] - 9)
#     plt.plot(a,t/b,'r',alpha=(float(i)+1)/(len(tau)))
#
# plt.show()

treshape=[]
plt.plot(np.repeat(0, len(age)), age, 'o')
for i in range(len(tau)):
    perc = np.zeros(len(a) - 1)
    dage = np.zeros(len(a) - 1)
    avgage = np.zeros(len(a))
    aa=np.linspace(0,1000,1000)
    scale = np.trapz(np.exp(-aa / np.power(10, tau[i] - 9)), aa)
    sfh = np.exp(-a / np.power(10, tau[i] - 9)) / scale
    for ii in range(len(a)-1):
        dage[0:ii+1]=dage[0:ii+1]+a[ii+1]-a[ii]
        perc[ii] = np.trapz(sfh[ii:ii + 2], a[ii:ii + 2])
        avgage[ii+1]=sum(dage*perc)/sum(perc)
    Avage=interp1d(a,avgage)(age)
    treshape.append(np.round(Avage,4))
    plt.plot(np.repeat(np.power(10, tau[i] - 9), len(age)), Avage, 'o')
    # plt.plot(np.repeat(tau[i], len(age)), Avage, 'o')
    # plt.plot(a,avgage,'r',alpha=(float(i)+1)/(len(tau)))
plt.tick_params(axis='both', which='major', labelsize=17)
plt.minorticks_on()
plt.gcf().subplots_adjust(bottom=0.16)
plt.xlabel('$\\tau$ (Gyrs)',size=20)
plt.ylabel('Average Age',size=20)
plt.show()

# # print treshape
#
# dat=Table(treshape,names=np.array(tau).astype(str))
# ascii.write(dat,'tau_scale_nage.dat')

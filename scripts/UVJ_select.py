from vtl.Readfile import Readfile
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table

def Mag(band):
    magnitude=25-2.5*np.log10(band)
    return magnitude

cat_data=Table.read('../../../Clear_data/goodsn_3dhst.v4.3.cat', format='ascii')
IDS=np.array(cat_data.field('id'))
ra=np.array(cat_data.field('ra'))
dec=np.array(cat_data.field('dec'))
J=np.array(cat_data.field('f_F125W'))
star=np.array(cat_data.field('class_star'))

fast_data=fits.open('../../../Desktop/catalogs_for_CLEAR/goodsn_3dhst.v4.1.cats/Fast/goodsn_3dhst.v4.1.fout.FITS')[1].data
lmass=np.array(fast_data.field('lmass'))


RF_dat=Readfile('../../../Desktop/catalogs_for_CLEAR/goodsn_3dhst.v4.1.cats/RF_colors/goodsn_3dhst.v4.1.master.RF',27)
ids=np.array(RF_dat[0])
z=np.array(RF_dat[1])
u=np.array(RF_dat[3])
v=np.array(RF_dat[7])
j=np.array(RF_dat[9])

######Get colors

uv=Mag(u)-Mag(v)
vj=Mag(v)-Mag(j)


print len(IDS)-len(ids)

INDEX1=[]

for i in range(len(IDS)):
    if J[i]!=-99 and star[i]<0.8 and 1<z[i]<2 and lmass[i] > 10:
        INDEX1.append(i)

INDEXQ=[]

for i in INDEX1:
    if uv[i]>=0.88*vj[i]+0.59 and uv[i]>1.382 and vj[i]<1.65:
        INDEXQ.append(i)

plt.plot(Mag(J[INDEXQ]), lmass[INDEXQ],'o')
plt.axvline(23)
plt.axhline(10)
plt.axhline(10.5)
plt.axhline(11)
plt.xlim(30,18)
plt.ylim(9,12)
plt.xlabel('J (Mag)',size=20)
plt.ylabel('log(M)',size=20)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.show()

data=Table([IDS[INDEXQ], ra[INDEXQ],dec[INDEXQ]], names=['ID','ra','dec'])
print data
# ascii.write(data,'UVJ_goodsn_updated.dat')
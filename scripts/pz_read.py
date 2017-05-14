import numpy as np
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table

Pz=np.loadtxt('Pofz_2.dat',delimiter=',')

# fn,zsmax,zschi,zs,zsel,zseh, zpspec_max ,zpspec_chi ,zpspec ,zpspec_el ,zpspec_eh=np.array(Readfile('stack_redshifts.dat',1,is_float=False))

fn=['s36095','s39825','s47677']

idx=np.zeros(len(fn))
for i in range(len(fn)):
    idx[i]=int(int(fn[i][1:])-1)
    print id
    # dat=Table(Pz[idx[i]],names='P(z)')
    # ascii.write(dat,'Pofz/%s_pofz.dat' % (idx[i]+1))
print Pz[idx[0]]
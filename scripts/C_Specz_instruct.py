from C_Specz import Specz_fit
import numpy as np
import sys

if __name__ == '__main__':
   galaxy = sys.argv[1] 

metal=np.round(np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]),3)
age=np.round(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),2)

z=np.round(np.arange(0.8,2.001,.001))

Specz_fit(metal,age,z,galaxy,'{0}_zfit'.format(galaxy))
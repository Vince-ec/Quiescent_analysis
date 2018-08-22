#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from C_MC_fit import MC_fit
import numpy as np
import sys

if __name__ == '__main__':
    galaxy = sys.argv[1] 
    specz = float(sys.argv[2])
    zlim1 = float(sys.argv[3])
    zlim2 = float(sys.argv[4])
    msim = float(sys.argv[5])
    asim = float(sys.argv[6])
    dataset = sys.argv[7]
    name = sys.argv[8]

metal=np.round(np.arange(0.002,0.031,0.001),3)
age=np.round(np.arange(.5,6.1,.1),1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
rshift = np.round(np.arange(zlim1, zlim2, 0.001),3)
sn = [2,4,8,12,16]


for i in range(len(sn)):
    MC_fit(galaxy, metal, age, tau, rshift, np.arange(0, 1.1, 0.1), msim, asim, 0, 
       specz, 0.0, sn[i], dataset, specz, name + '_' + str(sn[i]), repeats=1000)

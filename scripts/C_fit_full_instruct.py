#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from C_fit_full import Galaxy_full_fit
import numpy as np
import sys

if __name__ == '__main__':
    galaxy = sys.argv[1] 
    specz = float(sys.argv[2])
    zlim1 = float(sys.argv[3])
    zlim2 = float(sys.argv[4])
    name = sys.argv[5]

metal=np.round(np.arange(0.002,0.031,0.001),3)
age=np.round(np.arange(.5,6.1,.1),1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
rshift = np.round(np.arange(zlim1, zlim2, 0.001),3)

Galaxy_full_fit(metal, age, tau, rshift, specz, galaxy, name)

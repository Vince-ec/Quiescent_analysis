from spec_id import Gen_sim,MC_fit_methods_test_2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sea

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
# tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
#      9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
#
# metal=[0.005,0.015,0.02,0.025]
# age=[2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8]
# tau=[0,8.0, 8.3, 8.48, 8.6, 8.7]
tau=[0,8.0]

MC_fit_methods_test_2('n21156',metal,age,tau,0.019,3.5,0,1.251,repeats=100)
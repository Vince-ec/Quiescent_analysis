from spec_id import Nirspec_fit,MC_fit_jwst
import numpy as np

age = np.arange(0.5,2.1,0.1)
metal = np.arange(0.004,0.028,0.001)
tau = [0, 8.0, 8.48, 8.7, 8.85, 8.95, 9.04, 9.11, 9.18, 9.23, 9.28, 9.32, 9.36, 9.4, 9.43, 9.46]

# Nirspec_fit('../data/nirspec_sim_2.npy',metal,age,tau,'z3.717_2')
MC_fit_jwst('../data/nirspec_sim_2.npy',metal,age,tau,'z3.717_2',repeats=1000)

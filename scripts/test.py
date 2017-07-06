from spec_id import MC_fit
import numpy as np

metal=np.array([0.015,0.018,0.02,.025])
age=np.array([1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4])
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7]

MC_fit('s40862',metal,age,tau,0.015,3.2,8.6,1.328,'test',repeats=10)
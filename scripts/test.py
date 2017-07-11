from spec_id import MC_fit_methods
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

metal=np.array([0.015,0.018,0.02,.025])
age=np.array([1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4])
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7]

# MC_fit_methods('s39170',metal,age,tau,0.019,3.0,8.7,1.022,'s39170_test',maxwv=11400,repeats=50)

m,a=np.load('../mcerr/s39170_test.npy')
m_mn,a_mn=np.load('../mcerr/s39170_testmean.npy')
mnc,anc=np.load('../mcerr/s39170_testNC.npy')
mnc_mn,anc_mn=np.load('../mcerr/s39170_testNCmean.npy')
mdf,adf=np.load('../mcerr/s39170_testDF.npy')
mdf_mn,adf_mn=np.load('../mcerr/s39170_testDFmean.npy')


sea.kdeplot(m,a)
plt.scatter(m,a,marker='o')
plt.title('full fit')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()

sea.kdeplot(m_mn,a_mn)
plt.scatter(m_mn,a_mn,marker='o')
plt.title('full fit mean')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()

sea.kdeplot(mnc,anc)
plt.scatter(mnc,anc,marker='o')
plt.title('NC fit')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()

sea.kdeplot(mnc_mn,anc_mn)
plt.scatter(mnc_mn,anc_mn,marker='o')
plt.title('NC fit mean')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()

sea.kdeplot(mdf,adf)
plt.scatter(mdf,adf,marker='o')
plt.title('DF fit')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()

sea.kdeplot(mdf_mn,adf_mn)
plt.scatter(mdf_mn,adf_mn,marker='o')
plt.title('DF fit mean')
plt.scatter(0.019,3.0,color='r')
plt.axis([0,0.03,0,6])
plt.show()


import numpy as np
from scipy.interpolate import interp1d

def Intersect(array1,array2,independent,lowerbound,upperbound):
    diff=np.subtract(1,np.divide(array1,array2))
    rng=np.linspace(lowerbound,upperbound,1000)
    eq1=interp1d(independent,diff)
    eq2=np.polyfit(rng,eq1(rng),1)
    intersect=np.roots(eq2)
    test=np.add(np.multiply(eq2[0],rng),eq2[1])

    plt.plot(rng,eq1(rng))
    plt.plot(rng,test)
    plt.show()

    return intersect

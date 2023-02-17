# author: Yunxiao Zhang
# email: yunxiao9277@gmail.com
# date: 2020.12.2

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin
from scipy.integrate import quad
from scipy.optimize import fmin

# 01 vector norm
def norm(x,p):
    """
    ||x||_p
    Lp-norm
    """    
    if p == "inf":
        "infinity norm, Chebyshev norm, max norm"
        return max(np.abs(x))
    elif p == 1:
        "L1 norm, Taxicab norm, Manhattan norm"
        return np.sum(np.abs(x))
    elif p == 2:
        "L2 norm, Euclidean norm"
        return (np.sum(np.array(x)**2))**(1/2)
    elif p > 1 and p != 2:
        return np.sum(np.abs(x)**p)**(1/p)

    else:
        print("Wrong norm parameter! using L2 norm")
        return (np.sum(np.array(x)**2))**(1/2)
        

# 02 function norm
def norm_f(f,a,b,p):
    """
    ||x||_p
    p >=1
    Lp-norm
    1-norm = sum{i=1,n}{|x_i|}
    """    
    if p == "inf":
        fp = lambda x: np.abs(f(x))
        return fmin(fp)
    fp = lambda x: np.abs(f(x))**p
    return quad(fp,a,b)[0]**(1/p)

# 03 test
if __name__ == "__main__":
    if 0:
        x = [0.2,0.6, 32, 32]
        p_list = np.arange(0.1,10,0.1)
        norm_list = [norm(x,p) for p in p_list]
        plt.plot(p_list, norm_list)

        p_list = [1,2,10]
        norm_list = [norm(x,p) for p in p_list]

        plt.plot(p_list, norm_list, 'ro')
        plt.show()


    if 0:
        f = lambda x: x**2

        p_list = np.arange(0.1,10,0.1)
        norm_list = [norm_f(f,3,5,p) for p in p_list]
        plt.plot(p_list, norm_list)

        p_list = [0.1,1,10]
        norm_list = [norm_f(f,3,5,p) for p in p_list]
        plt.plot(p_list, norm_list, 'ro')

        plt.show()



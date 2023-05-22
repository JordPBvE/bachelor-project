import numpy as np
import matplotlib.pyplot as plt
import math
from classes import *

def COS_Error(option, value, time, N, previousextreme=None):
    estimate = COS_Estimate(option, value, time, N, previousextreme)

    if isinstance(option, European):
        analytic = option.analytic(value, time)
    elif isinstance(option, Lookback):
        analytic = option.analytic(value, time, previousextreme)

    return abs(estimate - analytic)

def COS_Estimate(option, value, time, N, previousextreme=None):
    tau = option.T - time
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = (2.0 / (option.b - option.a)) * np.real(option.phi_adjlog_from(us, value, time) * np.exp(-i * us * option.a))

    # defining the array of H_k values according to the option specific function
    if isinstance(option, European):
        Hs = np.array([option.H(k) for k in ks])
    elif isinstance(option, Lookback):
        Hs = np.array([option.H(k, previousextreme) for k in ks])

    # multiplying the Fk values by the Hk values and discounting term
    vs = np.array([np.exp(-option.gbm.mu * tau) * F * H for F, H in zip(Fks, Hs)])

    return sump(vs)

def COS_Density(option, value, time, N, y):
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = (2.0 / (option.b - option.a)) * np.real(option.phi_adjlog_from(us, value, time) * np.exp(-i * us * option.a))
    Fks[0] *= 0.5
    
    # defining array of cos(u*(y-a))
    cs = np.cos(np.outer(us, y - option.a))

    # inproduct of F_ks and cs => desired function
    f_X = np.matmul(Fks, cs)

    return f_X

def COS_Plot(option, value, time):
    y = np.linspace(0.05, 5.0, 1000)
    
    # plotting the resulting density function for multiple values of N
    for n in [2**k for k in range(2, 7)]:
        f_X = COS_Density(option, value, time, n, y)
        plt.plot(y, f_X)

    plt.savefig('./option_pricing/fig.png')

T = 1

gbm = GBM(mu=0, sigma=1)
lower, upper = -10, 10

eu = European(gbmprocess=gbm,
              exercisetime=T, 
              strikeprice=80,
              optionvariant=CALL, 
              lowerintegration=lower, 
              upperintegration=upper)


COS_Plot(eu, 100, 0)

analytic  = eu.analytic(100, 0)
cosmethod = COS_Estimate(eu, 100, 0, 256)
print(f'analytic:  {analytic} \ncosmethod: {cosmethod}')

# for n in [16, 32, 64, 128, 256]:
#     err = COS_Error(eu, 100, 0, n)
#     print(f'for N={n}, the error is {err}')
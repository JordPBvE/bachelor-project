import numpy as np
import matplotlib.pyplot as plt
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
    Fks = np.real(option.phi_adjlog_from(us, value, time) * np.exp(-i * us * option.a))
    Fks[0] *= 0.5

    # defining the array of H_k values according to the option specific function
    if isinstance(option, European):
        Hs = np.array([option.H(k) for k in ks])
    elif isinstance(option, Lookback):
        Hs = np.array([option.H(k, previousextreme) for k in ks])

    # multiplying the Fk values by the Hk values
    vs = np.array([F * H for F, H in zip(Fks, Hs)])

    return np.exp(-option.gbm.mu * tau) * np.sum(vs)

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

def EU(t, St, K, variant, T, mu, sigma):
    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(T-t)
    upper =  10*np.sqrt(T-t)

    eu = European(gbmprocess=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant, 
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    analytic = eu.analytic(St, t)
    print(f'analytic:    {analytic}')

    for n in [16, 32, 64, 128, 256]:
        est = COS_Estimate(eu, St, t, n)
        print(f'for N={n}, the estimate is {est}')
    
    # COS_Plot(eu, 100, 0)

def LB(t, St, K, prevex, variant, T, mu, sigma):

    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(2*T)
    upper =  10*np.sqrt(2*T)

    lb = Lookback(gbmprocess=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant,
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    analytic = lb.analytic(St, t, prevex)
    print(f'analytic:    {analytic}')

    for n in [16, 32, 64, 128, 256, 512, 1024]:
        est = COS_Estimate(lb, St, t, n, prevex)
        print(f'for N={n}, the estimate is {est}')
    
    # COS_Plot(lb, 100, 0)

# EU(t=0.1, St=100, K=120, variant=CALL, T=0.3, mu=0.05, sigma=0.2)

LB(t=0.1, St=100, K=120, prevex=130, variant=CALL, T=0.3, mu=0.05, sigma=0.2)
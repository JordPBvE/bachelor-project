import numpy as np
import matplotlib.pyplot as plt
from classes import *
from colorama import Fore, Style
np.set_printoptions(threshold=np.inf)

# function returning a cos method pdf estimate for given parameters
def COS_Estimate(option, value, time, N, previousextreme=None, filter=None):
    tau = option.T - time
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = Filter.filterterm(filter, ks, N) * np.real(option.phi_adjlog_from(us, value, time) * np.exp(-i * us * option.a))
    Fks[0] *= 0.5

    # defining the array of H_k values according to the option specific function
    Hs = np.array([option.H(k, previousextreme) for k in ks])

    # multiplying the Fk values by the Hk values
    vs = np.array([F * H for F, H in zip(Fks, Hs)])

    return np.exp(-option.process.mu * tau) * np.sum(vs)

# function returning the cos density estimate (vectorised)
def COS_Density(option, value, time, N, y, filter):
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = (2.0 / (option.b - option.a)) * Filter.filterterm(filter, ks, N) * np.real(option.phi_adjlog_from(us, value, time) * np.exp(-i * us * option.a))
    Fks[0] *= 0.5
    
    # defining array of cos(u*(y-a))
    cs = np.cos(np.outer(us, y - option.a))

    # inproduct of F_ks and cs => desired function
    f_X = np.matmul(Fks, cs)

    return f_X

# function plotting the output of COS_Density() + monte carlo for comparison
def COS_Plot(option : Option, value, time, ns, filter):
    y = np.linspace(-.5, .5, 1000)

    # plotting the resulting density function for multiple values of N
    for n in ns:
        f_X = COS_Density(option, value, time, n, y, filter)
        # print(max(f_X))
        plt.plot(y, f_X, label=f'N={n}')
        # plt.vlines(np.log(value/option.strike), -.5, 4, colors='k', linestyles='dotted')

    values = option.MonteCarlo(10000, 1000, time, value, prevex=value, adjlogs=True)
    plt.hist(values, density=True, bins=40)
    plt.grid()
    # plt.legend()
    plt.ylim(-1, 13)
    plt.savefig('./option_pricing_extended/fig.png')

    plt.figure()
    plt.grid()
    xs = np.linspace(-1, 1, 100)
    plt.plot(xs, [Filter.filterterm(filter, x, 1) for x in xs])
    plt.savefig('./option_pricing_extended/filter.png')

# Test function for lookback options
def LB(t, St, K, prevex, variant, T, mu, sigma, filter):
    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(T)
    upper =  10*np.sqrt(T)

    # defining the option parameters
    lb = Lookback(stockmechanic=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant,
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    # generating analytical solution
    analytic = lb.analytic(St, t, prevex)
    print(f'analytic:    {analytic}')

    # generating cos method value estimate for given N_cos values
    ns = [1024]
    for n in ns:
        est = COS_Estimate(lb, St, t, n, prevex, filter)
        print(f'for N={n}, the estimate is {est}')
    
    COS_Plot(lb, St, t, ns, filter)  

# function for comparing numerical and analytic a_k values
def ak_NUMvsANA(k, alpha, stepsize, T, lowerintegration, upperintegration, h):
    xmin = -20*math.sqrt(T)
    xmax =  20*math.sqrt(T)
    zs = np.linspace(xmin, xmax, 1000)

    gbm = GBM(0.1, 0.05)
    ana_a = lambda z : gbm.ana_a(k, z, 0, stepsize)
    num_a = lambda z : gbm.num_a(k, z, alpha, stepsize, lowerintegration, upperintegration, h)

    plt.figure()
    plt.grid()
    plt.plot(zs, [ana_a(z) for z in zs], label="analytic")
    plt.plot(zs, [num_a(z) for z in zs], label="numerical")
    plt.legend()

    plt.savefig('./option_pricing_extended/fig.png')

LB(t=0.5, St=100, prevex=110, K=100, variant=CALL, T=1, mu=0.05, sigma=0.2, filter=COS)

# ak_NUMvsANA(1000, 5, 0.001, 1, -10, 10, 1/10)

winsound.Beep(1000, 300)
winsound.Beep(1800, 1000)

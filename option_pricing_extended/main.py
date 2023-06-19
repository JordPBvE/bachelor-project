import numpy as np
import matplotlib.pyplot as plt
from classes import *
from colorama import Fore, Style

def COS_Estimate(option, value, time, N, previousextreme=None, filter=None):
    tau = option.T - time
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = np.array([Filter.filterterm(filter, k, N) * np.real(option.phi_adjlog_from(u, value, time) * np.exp(-i * u * option.a)) for k, u in zip(ks, us)])
    Fks[0] *= 0.5

    # defining the array of H_k values according to the option specific function
    Hs = np.array([option.H(k, previousextreme) for k in ks])

    # multiplying the Fk values by the Hk values
    vs = np.array([F * H for F, H in zip(Fks, Hs)])

    return np.exp(-option.process.mu * tau) * np.sum(vs)

def COS_Density(option, value, time, N, y, filter):
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = np.array([(2.0 / (option.b - option.a)) * Filter.filterterm(filter, k, N) * np.real(option.phi_adjlog_from(u, value, time) * np.exp(-i * u * option.a)) for k, u in zip(ks, us)])
    Fks[0] *= 0.5
    
    # defining array of cos(u*(y-a))
    cs = np.cos(np.outer(us, y - option.a))

    # inproduct of F_ks and cs => desired function
    f_X = np.matmul(Fks, cs)

    return f_X

def COS_Plot(option : Option, value, time, ns, filter):
    y = np.linspace(-1, 1, 200)

    # plotting the resulting density function for multiple values of N
    for n in ns:
        f_X = COS_Density(option, value, time, n, y, filter)
        plt.plot(y, f_X, label=f'N={n}')
        # plt.vlines(np.log(value/option.strike), -.5, 4, colors='k', linestyles='dotted')

    values = option.MonteCarlo(10000, 1000, time, value, prevex=value, adjlogs=True)
    plt.hist(values, density=True, bins=40)
    plt.grid()
    plt.legend()
    plt.savefig('./option_pricing_extended/fig.png')

    plt.figure()
    plt.grid()
    xs = np.linspace(-1, 1, 100)
    plt.plot(xs, [Filter.filterterm(filter, x, 1) for x in xs])
    plt.savefig('./option_pricing_extended/filter.png')

def LB(t, St, K, prevex, variant, T, mu, sigma, filter):
    gbm = GBM(mu, sigma)
    lower = -15*np.sqrt(T)
    upper =  15*np.sqrt(T)

    lb = Lookback(stockmechanic=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant,
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    analytic = lb.analytic(St, t, prevex)
    # montecarlo = lb.MonteCarlo(10000, int(200 / T), t, St, prevex)
    print(f'analytic:    {analytic}')
    # print(f'monte carlo: {montecarlo}\n')
    ns = []
    for n in ns:
        est = COS_Estimate(lb, St, t, n, prevex, filter)
        print(f'for N={n}, the estimate is {est}')
    
    COS_Plot(lb, St, t, [1024], filter)

def ak_NUMvsANA(k, alpha, stepsize, T, lowerintegration, upperintegration, h):
    xmin = -10*math.sqrt(T)
    xmax =  10*math.sqrt(T)

    zs = np.linspace(xmin, xmax, 1000)

    ana_a = lambda z : gbm.ana_a(k, z, stepsize)
    num_a = lambda z : gbm.num_a(k, z, alpha, stepsize, lowerintegration, upperintegration, h)


    plt.figure()
    plt.grid()
    plt.plot(zs, [ana_a(z) for z in zs], label="analytic")
    plt.plot(zs, [num_a(z) for z in zs], label="numerical")
    plt.legend()

    plt.savefig('./option_pricing_extended/fig.png')

def phimax_comparison():
    gbm = GBM(0.1, 0.2)
    m = 200
    x = 100
    t = 0
    T = 1
    K = 100

    xmin = -10*math.sqrt(T)
    xmax =  10*math.sqrt(T)

    zs = np.linspace(xmin, xmax, 50)

    plt.grid()
    plt.plot(zs, [gbm.phi_adjlogmax(z, x, t, T, K) for z in zs], label='analytic')
    plt.plot(zs, [gbm.spitzer_result(m, z, x, t, T, K, num=False) for z in zs], label="nspitzer opt")
    plt.plot(zs, [gbm.spitzer_result2(m, z, x, t, T, K, num=False) for z in zs], label="nspitzer opt")
    # plt.plot(zs, [gbm.slow_spitzer_result(m, z, x, t, T, K) for z in zs], label="spitzer slow")
    plt.legend()

    plt.savefig('./option_pricing_extended/phimax.png')

LB(t=0.5, St=100, prevex=110, K=120, variant=CALL, T=1, mu=0.05, sigma=0.2, filter=None)

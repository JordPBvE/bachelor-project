import numpy as np
import matplotlib.pyplot as plt
from classes import *
from colorama import Fore, Style

def MonteCarloDemo():
    print('\t|\t|          T=1/4\t|          T=1/2\t|          T=1\t\t|          T=2\t\t|')
    print(f'\t|   K\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|')
    for strike in [9, 10, 11]:
        for var in [CALL, PUT]:
            print(f'{var.value.lower()}\t| {strike}\t', end='', flush=True)
            for T in [1/4, 1/2, 1, 2]:
                gbm = GBM(0.1, 0.4)
                lower = -10*np.sqrt(2*T)
                upper =  10*np.sqrt(2*T)

                t = 0.2
                St = 9
                if var == CALL : prevex = 10
                if var == PUT  : prevex = 8

                lb = Lookback(gbmprocess=gbm,
                            exercisetime=T, 
                            strikeprice=strike,
                            optionvariant=var,
                            lowerintegration=lower, 
                            upperintegration=upper)
                analytic   = round(lb.analytic(St, t, prevex), 2)
                montecarlo = round(lb.MonteCarlo(10000, 1000, t, St, prevex), 2)
                error      = round(abs(analytic - montecarlo), 2)
                out = ''
                out += f'| {"{:.2f}".format(analytic)} \t'
                out += f'| {"{:.2f}".format(montecarlo)}\t'
                out += f'| {Fore.RED}{str(error)}{Style.RESET_ALL}\t'
                print(out, end='', flush=True)
            print('|', flush=True)

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
    Fks = np.array([np.real(option.phi_adjlog_from(u, value, time) * np.exp(-i * u * option.a)) for u in us])
    Fks[0] *= 0.5

    # defining the array of H_k values according to the option specific function
    if isinstance(option, European):
        Hs = np.array([option.H(k) for k in ks])
    elif isinstance(option, Lookback):
        Hs = np.array([option.H(k, previousextreme) for k in ks])

    # multiplying the Fk values by the Hk values
    vs = np.array([F * H for F, H in zip(Fks, Hs)])

    return np.exp(-option.process.mu * tau) * np.sum(vs)

def COS_Density(option, value, time, N, y):
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (option.b - option.a)  # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    Fks = np.array([(2.0 / (option.b - option.a)) * np.real(option.phi_adjlog_from(u, value, time) * np.exp(-i * u * option.a)) for u in us])
    Fks[0] *= 0.5
    
    # defining array of cos(u*(y-a))
    cs = np.cos(np.outer(us, y - option.a))

    # inproduct of F_ks and cs => desired function
    f_X = np.matmul(Fks, cs)

    return f_X

def COS_Plot(option : Option, value, time, ns):
    y = np.linspace(-1, 1, 1000)
    
    # plotting the resulting density function for multiple values of N
    for n in ns:
        f_X = COS_Density(option, value, time, n, y)
        plt.plot(y, f_X)
        # plt.vlines(np.log(value/option.strike), -.5, 4, colors='k', linestyles='dotted')

    values = option.MonteCarlo(10000, 1000, time, value, prevex=value, adjlogs=True)

    plt.hist(values, density=True, bins=40)

    plt.savefig('./option_pricing_extended/fig.png')

def EU(t, St, K, variant, T, mu, sigma):
    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(T-t)
    upper =  10*np.sqrt(T-t)

    eu = European(stockmechanic=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant, 
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    analytic   = eu.analytic(St, t)
    # montecarlo = eu.MonteCarlo(10000, 1000, t, St)
    print(f'analytic:    {analytic}')
    # print(f'monte carlo: {montecarlo}\n')

    ns = [256]
    for n in ns:
        est = COS_Estimate(eu, St, t, n)
        print(f'for N={n}, the estimate is {est}')
    
    # COS_Plot(eu, St, t, ns)

def LB(t, St, K, prevex, variant, T, mu, sigma):
    print(f'======================{variant.value.upper()}======================')
    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(T)
    upper =  10*np.sqrt(T)

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
    ns = [64, 128, 256]
    for n in ns:
        est = COS_Estimate(lb, St, t, n, prevex)
        print(f'for N={n}, the estimate is {est}')
    
    # COS_Plot(lb, St, t, [256])

EU(t=0.5, St=100, K=80, variant=CALL, T=1, mu=0.1, sigma=0.2)
LB(t=0.1, St=100, prevex=110, K=100, variant=CALL, T=1/2, mu=0.1, sigma=0.2)
LB(t=0.1, St=100, prevex=100, K=100, variant=PUT, T=1/2, mu=0.05, sigma=0.2)

# MonteCarloDemo()
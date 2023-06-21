import numpy as np
import matplotlib.pyplot as plt
from classes import *
from colorama import Fore, Style
import time

# function verifying analytical solutions for lookback options via monte carlo estaimtes
def MonteCarloDemo():
    print('\t|\t|          T=1/4\t|          T=1/2\t|          T=1\t\t|          T=2\t\t|')
    print(f'\t|   K\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|  ANA\t|   MC\t|  {Fore.RED}ERR{Style.RESET_ALL}\t|')
    # for loops for different option parameters
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

                # defining the option
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

# function returning the error a particular cos estimate makes
def COS_Error(option, value, time, N, previousextreme=None):
    estimate = COS_Estimate(option, value, time, N, previousextreme)

    if isinstance(option, European):
        analytic = option.analytic(value, time)
    elif isinstance(option, Lookback):
        analytic = option.analytic(value, time, previousextreme)

    return abs(estimate - analytic)

# function returning a cos method pdf estimate for given parameters
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

    # multiplying the Fk values by the Hk values to give cos method sum
    vs = np.array([F * H for F, H in zip(Fks, Hs)])

    return np.exp(-option.process.mu * tau) * np.sum(vs)

# function returning the cos density estimate (vectorised)
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

# function plotting the output of COS_Density() + monte carlo for comparison
def COS_Plot(option : Option, value, time, ns):
    y = np.linspace(-1, 1, 1000)
    
    # plotting the resulting density function for multiple values of N
    for n in ns:
        f_X = COS_Density(option, value, time, n, y)
        plt.plot(y, f_X, label=f'N={n}')

    values = option.MonteCarlo(10000, 1000, time, value, prevex=value, adjlogs=True)
    plt.hist(values, density=True, bins=40)
    # plt.legend()
    plt.grid()
    plt.savefig('./option_pricing/fig.png')

# Test function for european options
def EU(t, St, K, variant, T, mu, sigma):
    gbm = GBM(mu, sigma)
    lower = -10*np.sqrt(T-t)
    upper =  10*np.sqrt(T-t)

    # defining the option parameters
    eu = European(stockmechanic=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant, 
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    # generating analytical solution and monte carlo estimate
    analytic   = eu.analytic(St, t)
    t_start    = time.time()
    montecarlo = eu.MonteCarlo(10000, 1000, t, St)
    t_stop     = time.time()
    print(f'analytic:    {analytic}')
    print(f'monte carlo: {"{:.2e}".format(montecarlo - analytic)}, this took {round(1000*(t_stop - t_start), 2)} msec\n')

    # generating cos method value error and duration for given N_cos values
    ns = [256]
    for n in ns:
        t_start = time.time()
        err = "{:.2e}".format(abs(COS_Estimate(eu, St, t, n) - analytic))
        t_stop  = time.time()
        print(f'N={n},\t err={err},\t this took {round(1000*(t_stop - t_start), 3)} msec')
    
    # plotting the estimated pdf for given N_cos values
    COS_Plot(eu, St, t, ns)

# Test function for lookback options
def LB(t, St, K, prevex, variant, T, mu, sigma):
    gbm = GBM(mu, sigma)
    lower =  np.log(St/K)
    lower =  -10*np.sqrt(T)
    upper =  10*np.sqrt(T)

    # defining the option parameters
    lb = Lookback(stockmechanic=gbm,
                  exercisetime=T, 
                  strikeprice=K,
                  optionvariant=variant,
                  lowerintegration=lower, 
                  upperintegration=upper)
    
    # generating analytical solution and monte carlo estimate
    analytic   = round(lb.analytic(St, t, prevex), 6)
    t_start = time.time()
    montecarlo = round(lb.MonteCarlo(10000, 1000, t, St, prevex), 2)
    t_stop  = time.time()
    print(f'analytic:    {analytic}')
    print(f'monte carlo: {"{:.2e}".format(abs(montecarlo - analytic))}, this took {round(1000*(t_stop - t_start), 2)} msec\n')

    # generating cos method value error and duration for given N_cos values
    ns = [256]
    for n in ns:
        t_start = time.time()
        err = "{:.2e}".format(abs(COS_Estimate(lb, St, t, n, prevex) - analytic))
        t_stop  = time.time()
        print(f'N={n},\t err={err},\t this took {round(1000*(t_stop - t_start), 5)} msec')
    
    # plotting the estimated pdf for given N_cos values
    COS_Plot(lb, St, t, ns)

EU(t=0.5, St=100, K=100, variant=CALL, T=1, mu=0.1, sigma=0.4)
# LB(t=0.5, St=100, prevex=110, K=100, variant=CALL, T=1, mu=0.1, sigma=0.4)

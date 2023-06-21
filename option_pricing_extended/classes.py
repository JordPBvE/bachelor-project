import numpy as np
import math
import random
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from scipy.special import erfc
from scipy.stats import norm
import sys
from colorama import Fore, Style
import time
import winsound
np.seterr(all='raise')

# defining imaginary unit
i = complex(0.0, 1.0)

# custom class for option variants
class OptionVariant(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
PUT = OptionVariant.PUT
CALL = OptionVariant.CALL

# class defining filters and generating filter terms
class Filter(Enum):
    FEJER        = 'FEJER'
    RAISEDCOSINE = 'RAISEDCOSINE'
    EXPONENTIAL  = 'EXPONENTIAL'

    def filterterm(var, n, N):
        match var:
            case Filter.FEJER:
                out = 1 - abs(n/N)
            case Filter.RAISEDCOSINE:
                out = 0.5*(1 + np.cos(math.pi * n/N))
            case Filter.EXPONENTIAL:
                alpha = - np.log(2.22*10**(-16))
                p=4
                return np.exp(-alpha*(n/N)**p)
            case _:
                return 1
        return out
FEJ = Filter.FEJER
COS = Filter.RAISEDCOSINE
EXP = Filter.EXPONENTIAL

# sum prime operator from grzelak and oosterlee
def sump(array):
    if array.size == 0:
        return 0
    res = np.sum(array) - array[0]/2
    return res

# charactersitic function standard normal distribution
def phi_N(u):
    try:
        return np.exp(-0.5 * u**2)
    except:
        return 0

# parent class defining stochastic processes
class Process:
    # function plotting the integrand in numerical a_k calculation
    def plotintegrand(self, k, z, alpha, stepsize, lowerintegration, upperintegration, h):
        i = complex(0, 1)
        tau_k = k * stepsize
        xs = np.linspace(lowerintegration, upperintegration, 1000) + i * alpha
        f = lambda xi : self.phi_levy(tau_k, -xi)*z / (xi**2 + xi*z)

        plt.figure()
        plt.plot(xs, f(xs))
        xs = np.arange(lowerintegration, upperintegration + h, h) + i*alpha
        for i, x in enumerate(xs[:-1]):
            plt.plot([xs[i], xs[i+1]], [f(xs[i]), f(xs[i+1])], 'ko', linestyle='solid', linewidth=.7, markersize=2)


        plt.savefig('./option_pricing_extended/integrand.png')
    
    # function returning adjusted logged spitzer recurrence formula
    def spitzer_result(self, m : int, u : np.array, x, t, T, K, num=False):
        return np.exp(i*u*np.log(x/K)) * self.spitzer_recurrence(m, u, t, T, num=False)
        # return np.exp(i*u*np.log(x/K)) * (self.spitzer_recurrence(m, u, t, T, num=False) + self.spitzer_recurrence(m, -u, t, T, num=False))

    # spitzer recurrence algorithm from haslip and kaishev
    def spitzer_recurrence(self, m : int, u : np.array, t, T, num=False):
        p = np.zeros((m, m, u.size), dtype=complex)
        for I in range(m):
            p[0,I] = self.ana_a(I, u, t, (T-t)/m) / (I+1)

        out = p[0, m-1]
        
        for I in range(1, m):
            for J in range(m - I):
                for k in range(J+1):
                    try:
                        p[I, J] += p[I-1, k] * p[0, J-k]
                    except Exception as e:
                        print(p[I-1, k] * p[0, J-k])
                        raise e
            try:
                out += p[I, m-1-I] / math.factorial(I+1)
            except Exception as e:
                # print('spitzer recurrence error caught: ' + str(e))
                pass

        return out
    
    # function calculating a_k coefficient via trapezoid rule
    def num_a(self, k, z, alpha, stepsize, lowerintegration, upperintegration, h):
        tau_k = k * stepsize 

        xs = np.arange(lowerintegration, upperintegration + h, h) + i*alpha
        f = lambda xi : self.phi_levy(tau_k, -xi)*z / (xi**2 + xi*z)

        sum = 0
        for k, x in enumerate(xs[:-1]):
            sum += 0.5 * (f(xs[k]) + f(xs[k+1])) * h
        
        return 1 - (i / (2*math.pi)) * sum

# child class defining geometric brownian motion
class GBM(Process):
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma
    
    # defining the charactersitic function of gbm exponent
    def phi_levy(self, tau, z):
        return np.exp(i*z*self.mu*tau - 0.5*self.sigma**2*z**2*tau)

    # analytical a_k coefficients
    def ana_a(self, k, z, t, stepsize):
        try:
            tau_k = k * stepsize
            alpha = self.mu - self.sigma**2/2

            t1 = norm.cdf(-alpha*np.sqrt(tau_k)/(self.sigma))
            f1 = 0.5*np.exp(-0.5*z**2*self.sigma**2*tau_k + 1j*tau_k*z*alpha)
            f2 = erfc(-np.sqrt(tau_k/2) * (z*self.sigma*1j + alpha/self.sigma))
            return t1 + f1 * f2
        except FloatingPointError as e:
            # print('ana_a error caught: ' + str(e))
            return 0
        
# parent class defining options 
class Option:
    def __init__(self, stockmechanic, exercisetime, strikeprice, optionvariant, lowerintegration, upperintegration):
        self.process = stockmechanic
        self.mu      = stockmechanic.mu
        self.sigma   = stockmechanic.sigma
        self.T       = exercisetime
        self.strike  = strikeprice
        self.optvar  = optionvariant
        self.a       = lowerintegration
        self.b       = upperintegration

    # function used in H_k calculation
    def chi(self, k, c, d):
        return 1/(1 + (k * np.pi / (self.b-self.a))**2) * (np.cos(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d)
                                                         - np.cos(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c)
                                                         + np.sin(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d) * (k*np.pi) / (self.b-self.a)
                                                         - np.sin(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c) * (k*np.pi) / (self.b-self.a))

    # function used in H_k calculation
    def psi(self, k, c, d):
        if k == 0: 
            return (d-c)
        else:
            return (np.sin(k*np.pi*(d-self.a)/(self.b-self.a)) - np.sin(k*np.pi*(c-self.a)/(self.b-self.a))) * (self.b-self.a) / (k*np.pi)
    
    # function giving monte carlo estimate of option value (showing a progress percentage furing calculation)
    def MonteCarlo(self, iterations, steps, t, st, prevex=None, adjlogs = False):
        vs = np.zeros(iterations)

        percentage = 0.0
        print("{: >6.2f}".format(percentage) + '%', end='', flush=True)

        # monte carlo iterations loop
        for i in range(iterations):
            newperc = round(100*i/iterations, 2)
            if newperc != percentage:
                percentage = newperc
                sys.stdout.write('\b' * 7 + ' ' * 7 + '\b' * 7)
                print(f'{Fore.MAGENTA + Style.BRIGHT}{"{: >6.2f}".format(newperc)}%{Style.RESET_ALL}', end='', flush=True)
            
            # generating sample path and determining payoff
            gbm = self.samplepath(self.T - t, steps, st)
            vs[i] = self.payoff(gbm, prevex, adjlogs)
        sys.stdout.write('\b\b\b\b\b\b\b')


        # return all value endpoints for monte carlo graphing
        if adjlogs: return vs

        # returning monte carlo estimate
        else:       return np.exp(-self.mu*(self.T-t)) * np.average(vs)

    # function generating a gbm sample path
    def samplepath(self, tau, steps, st):
        srw = (1 / np.sqrt(steps)) * np.cumsum([0] + [random.choice([-1, 1]) for _ in range(int(steps * tau) - 1)])
        ts  = np.arange(int(steps * tau)) / steps

        return st * np.exp((self.mu - np.power(self.sigma, 2)/2) * ts + self.sigma * srw)      
    
# child class describing lookback option contracts
class Lookback(Option):
    # function determining payoff for a given path
    def payoff(self, path, prevex, adjlogs=False):
        mx = np.max(np.append(path, prevex))
        mn = np.min(np.append(path, prevex))

        if adjlogs:
            if self.optvar == CALL: return np.log(mx/self.strike)
            if self.optvar == PUT:  return np.log(mn/self.strike)

        if self.optvar == CALL: return np.maximum(mx - self.strike, 0) 
        if self.optvar == PUT:  return np.maximum(self.strike - mn, 0) 

    # determining payoff coefficients
    def H(self, k, prevex):
        adjprev = np.log(prevex/self.strike)

        tc1 = tc2 = tc3 = 0
        tp1 = tp2 = tp3 = 0
        
        if prevex > self.strike:
            tc1 = self.psi(k, self.a, adjprev) * 2*(prevex-self.strike)/(self.b-self.a)
            tc2 = (self.chi(k, adjprev, self.b) - self.psi(k, adjprev, self.b)) * 2*self.strike/(self.b-self.a)
            tp3 = (self.psi(k, self.a, 0) - self.chi(k, self.a, 0)) * 2*self.strike/(self.b-self.a)
        else:
            tc3 = (self.chi(k, 0, self.b) - self.psi(k, 0, self.b)) * 2*self.strike/(self.b-self.a)
            tp1 = self.psi(k, adjprev, self.b) * 2*(self.strike-prevex)/(self.b-self.a)
            tp2 = (self.psi(k, self.a, adjprev) - self.chi(k, self.a, adjprev)) * 2*self.strike/(self.b-self.a)

        if self.optvar is CALL: return tc1 + tc2 + tc3
        if self.optvar is PUT:  return tp1 + tp2 + tp3
    
    # funtion returning characteristic function for maximum process starting at a time t
    def phi_adjlog_from(self, u, x, t):
        m=20
        out = self.process.spitzer_result(m=m, u=u, x=x, t=t, T=self.T, K=self.strike, num=False)
        return out

    # funtion generating analytic solution  
    def analytic(self, x, t, prevex):
        tau = self.T - t
       
        tc1 = lambda input : x * norm.cdf(self.d(input, x, t)) - np.exp(-self.mu*tau) * input * norm.cdf(self.d(input, x, t) - self.sigma*math.sqrt(tau))
        tc2 = lambda input : np.exp(-self.mu*tau)*(self.sigma**2/(2*self.mu)) * x * (-(x/input)**(-(2*self.mu)/(self.sigma**2)) * norm.cdf(self.d(input, x, t) - 2*self.mu*math.sqrt(tau)/self.sigma) + np.exp(self.mu*tau)*norm.cdf(self.d(input, x, t)))

        tp1 = lambda input : -x * norm.cdf(-self.d(input, x, t)) + np.exp(-self.mu*tau) * input * norm.cdf(self.sigma*math.sqrt(tau) - self.d(input, x, t))
        tp2 = lambda input : np.exp(-self.mu*tau)*(self.sigma**2/(2*self.mu)) * x * ((x/input)**(-(2*self.mu)/(self.sigma**2)) * norm.cdf(2*self.mu*math.sqrt(tau)/self.sigma - self.d(input, x, t)) - np.exp(self.mu*tau)*norm.cdf(-self.d(input, x, t)))


        if self.optvar == CALL and prevex <  self.strike: return tc1(self.strike) + tc2(self.strike) 
        if self.optvar == CALL and prevex >= self.strike: return tc1(prevex) + tc2(prevex) + np.exp(-self.mu*tau)*(prevex-self.strike)

        if self.optvar == PUT  and prevex >= self.strike: return tp1(self.strike) + tp2(self.strike)
        if self.optvar == PUT  and prevex <  self.strike: return tp1(prevex) + tp2(prevex) + np.exp(-self.mu*tau)*(self.strike-prevex)
    
    # d_x coefficients used in analytic solution calculation
    def d(self, sub, x, t):
        tau = self.T - t
        return (np.log(x/sub) + self.mu*tau + 0.5*self.sigma**2*tau) / (self.sigma*math.sqrt(tau))
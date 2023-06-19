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
np.seterr(all='raise')

np.set_printoptions(linewidth=np.inf)
# GENERAL CONSTANTS AND GENERAL FUNCTIONS
i = complex(0.0, 1.0)

class OptionVariant(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
PUT = OptionVariant.PUT
CALL = OptionVariant.CALL

class Filter(Enum):
    FEJER        = 'FEJER'
    LANCZOS      = 'LANCZOS'
    RAISEDCOSINE = 'RAISEDCOSINE'
    EXPONENTIAL  = 'EXPONENTIAL'

    def filterterm(var, n, N):
        match var:
            case Filter.FEJER:
                out = 1 - abs(n/N)
            case Filter.LANCZOS:
                out = math.sin(math.pi * n/N) / (math.pi * n/N)
            case Filter.RAISEDCOSINE:
                out = 0.5*(1 + math.cos(math.pi * n/N))
            case Filter.EXPONENTIAL:
                alpha = - np.log(2.22*10**(-16))
                p=2
                return np.exp(-alpha*(n/N)**p)
            case _:
                return 1
        return out

FEJER        = Filter.FEJER
LANCZOS      = Filter.LANCZOS
RAISEDCOSINE = Filter.RAISEDCOSINE
EXPONENTIAL  = Filter.EXPONENTIAL

def sump(array):
    if array.size == 0:
        return 0
    res = np.sum(array) - array[0]/2
    return res

def phi_N(u):
    return np.exp(-0.5 * u**2)

def IaN(num):
    if not math.isnan(abs(num)): return num
    return 0

class Process:
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
    
    def slow_spitzer_result(self, m : int, u : complex, x, t, T, K):
        return np.exp(i*u*np.log(x/K)) * self.spitzer_recursion(m, (T-t)/m, u, t, T)
    
    def spitzer_recursion(self, k : int, stepsize, u : complex, t, T):
        if k == 0: return 1

        ts = [self.spitzer_recursion(j, stepsize, u, t, T) * self.ana_a(k-j, u, t, stepsize) for j in range(k)] 
        return np.sum(ts) / k 

    def spitzer_result(self, m : int, u : complex, x, t, T, K, num=False):
        return np.exp(i*u*np.log(x/K)) * self.spitzer_recurrence(m, u, x, t, T, K, num=False)
        return np.exp(i*u*np.log(x/K)) * (self.spitzer_recurrence(m, u, x, t, T, K, num=False) + self.spitzer_recurrence(m, -u, x, t, T, K, num=False))

    def spitzer_recurrence(self, m : int, u : complex, x, t, T, K, num=False):
        p = np.zeros((m, m), dtype=complex)
        for I in range(m):
            try:
                p[0,I] = self.ana_a(I, u, t, (T-t)/m) / (I+1)
            except:
                p[0,I] = 0
        
        out = p[0, m-1]
        
        for I in range(1, m):
            for J in range(m - I):
                for k in range(J+1):
                    p[I, J] += p[I-1, k] * p[0, J-k]
            try:
                out += p[I, m-1-I] / math.factorial(I+1)
            except:
                pass

        return IaN(out)
    
    def spitzer_result2(self, m : int, u : complex, x, t, T, K, num=False):
        p = np.zeros((m, m), dtype=complex)
        for I in range(m):
            try:
                p[0,I] = self.ana_a(I, u, t, (T-t)/m) / (I+1)
            except:
                p[0,I] = 0
        
        out = p[0, m-1]
        
        for I in range(1, m):
            for J in range(m - I):
                for k in range(J+1):
                    p[I, J] += p[I-1, k] * p[0, J-k]
            try:
                out += p[I, m-1-I] / math.factorial(I+1)
            except:
                pass

        return IaN(np.exp(i*u*np.log(x/K)) * out)
    
    def num_a(self, k, z, alpha, stepsize, lowerintegration, upperintegration, h):
        tau_k = k * stepsize 

        xs = np.arange(lowerintegration, upperintegration + h, h) + i*alpha
        f = lambda xi : self.phi_levy(tau_k, -xi)*z / (xi**2 + xi*z)

        sum = 0
        for k, x in enumerate(xs[:-1]):
            sum += 0.5 * (f(xs[k]) + f(xs[k+1])) * h
        
        return 1 - (i / (2*math.pi)) * sum

# CLASS DEFINING A GEOMETRIC BROWNIAN MOTION AND CHARACTERISTIC FUNCTIONS FOR PROCESSES DERIVED FROM GEOMETRIC BROWNIAN MOTIONS
class GBM(Process):
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma
    
    def phi_levy(self, tau, z):
        return np.exp(i*z*self.mu*tau - 0.5*self.sigma**2*z**2*tau)

    def ana_a(self, k, z, t, stepsize):
        tau_k = k * stepsize
        alpha = self.mu - self.sigma**2/2

        t1 = norm.cdf(-alpha*math.sqrt(tau_k)/(self.sigma))
        f1 = 0.5*np.exp(-0.5*z**2*self.sigma**2*tau_k + i*tau_k*z*alpha)
        f2 = erfc(-math.sqrt(tau_k/2) * (z*self.sigma*i + alpha/self.sigma))
        return IaN(t1 + f1 * f2)
    
    def phi_adjlogmax(self, u, x, t, T, K):
        tau = T - t
        alpha = (self.mu - 0.5*self.sigma**2) / self.sigma

        f1 = 2 * np.exp(i*u*np.log(x/K))
        f2 = 1 - (alpha / (i*self.sigma*u + 2*alpha))
        f3 = np.exp(-i * (self.sigma*u - 2*alpha*i) * tau * alpha)
        f4 = phi_N(math.sqrt(tau) * (self.sigma * u - 2*alpha*i))

        return f1 * f2 * f3 * f4
# PARENT CLASS DEFINING OPTIONS 
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

    def chi(self, k, c, d):
        return 1/(1 + (k * np.pi / (self.b-self.a))**2) * (np.cos(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d)
                                                         - np.cos(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c)
                                                         + np.sin(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d) * (k*np.pi) / (self.b-self.a)
                                                         - np.sin(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c) * (k*np.pi) / (self.b-self.a))

    def psi(self, k, c, d):
        if k == 0: 
            return (d-c)
        else:
            return (np.sin(k*np.pi*(d-self.a)/(self.b-self.a)) - np.sin(k*np.pi*(c-self.a)/(self.b-self.a))) * (self.b-self.a) / (k*np.pi)
        
    def MonteCarlo(self, iterations, steps, t, st, prevex=None, adjlogs = False):
        vs = np.zeros(iterations)

        percentage = 0.0
        print("{: >6.2f}".format(percentage) + '%', end='', flush=True)
        for i in range(iterations):
            newperc = round(100*i/iterations, 2)
            if newperc != percentage:
                percentage = newperc
                sys.stdout.write('\b' * 7 + ' ' * 7 + '\b' * 7)
                print(f'{Fore.MAGENTA + Style.BRIGHT}{"{: >6.2f}".format(newperc)}%{Style.RESET_ALL}', end='', flush=True)
            
            
            gbm = self.samplepath(self.T - t, steps, st)
            vs[i] = self.payoff(gbm, prevex, adjlogs)
        sys.stdout.write('\b\b\b\b\b\b\b')
        # print(vs)


        if adjlogs: return vs
        else:       return np.exp(-self.mu*(self.T-t)) * np.average(vs)

    def samplepath(self, tau, steps, st):
        srw = (1 / np.sqrt(steps)) * np.cumsum([0] + [random.choice([-1, 1]) for _ in range(int(steps * tau) - 1)])
        ts  = np.arange(int(steps * tau)) / steps

        return st * np.exp((self.mu - np.power(self.sigma, 2)/2) * ts + self.sigma * srw)      
    
# CHILD OPTION CLASS DESCRIBING COS METHOD PARAMETERS FOR LOOKBACK OPTIONS
class Lookback(Option):
    def payoff(self, path, prevex, adjlogs=False):
        mx = np.max(np.append(path, prevex))
        mn = np.min(np.append(path, prevex))

        if adjlogs:
            if self.optvar == CALL: return np.log(mx/self.strike)
            if self.optvar == PUT:  return np.log(mn/self.strike)

        if self.optvar == CALL: return np.maximum(mx - self.strike, 0) 
        if self.optvar == PUT:  return np.maximum(self.strike - mn, 0) 

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
       
    def phi_adjlog_from(self, u, x, t):
        # return self.process.phi_adjlogmax(u, x, t, self.T, self.strike)
        return self.process.spitzer_result(m=200, u=u, x=x, t=t, T=self.T, K=self.strike, num=False)
        
    
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
    
    def d(self, sub, x, t):
        tau = self.T - t
        return (np.log(x/sub) + self.mu*tau + 0.5*self.sigma**2*tau) / (self.sigma*math.sqrt(tau))
    
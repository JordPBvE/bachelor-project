import numpy as np
import math
import random
from enum import Enum
from scipy.stats import norm
import sys
from colorama import Fore, Style
np.seterr(all='raise')
np.set_printoptions(linewidth=np.inf)

# defining imaginary unit
i = complex(0.0, 1.0)

# custom class for option variants
class OptionVariant(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
PUT = OptionVariant.PUT
CALL = OptionVariant.CALL

# sum prime operator from grzelak and oosterlee
def sump(array):
    if array.size == 0:
        return 0
    res = np.sum(array) - array[0]/2
    return res

# charactersitic function standard normal distribution
def phi_N(u):
    return np.exp(-0.5 * u**2)

# class defining a geometric brownian motion and characteristic functions for processes derived from geometric brownian motions
class GBM:
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma
    
    # characteristic function for adjusted log gbm process
    def phi_adjlog(self, u, x, t, T, K):
        tau = T - t

        e1 = i*u*np.log(x/K)
        e2 = i*u*(self.mu - 0.5*self.sigma**2)*tau
        e3 = -0.5*tau*(self.sigma*u)**2

        return np.exp(e1 + e2 + e3)

    # characteristic function for adjusted log maximum gbm process
    def phi_adjlogmax(self, u, x, t, T, K):
        tau = T - t
        alpha = (self.mu - 0.5*self.sigma**2) / self.sigma

        f1 = 2 * np.exp(i*u*np.log(x/K))
        f2 = (1 - (alpha / (i*self.sigma*u + 2*alpha)))
        f3 = np.exp(-i * (self.sigma*u - 2*alpha*i) * tau * alpha)
        f4 = phi_N(math.sqrt(tau) * (self.sigma * u - 2*alpha*i))

        return f1 * f2 * f3 * f4
    
# parent class defining option contracts 
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
        if self.optvar is CALL: return self.process.phi_adjlogmax(u, x, t, self.T, self.strike)
        if self.optvar is PUT:  raise Exception("Lookback puts are not available")
    
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


# child class describing european option contracts
class European(Option):
    # function determining payoff for a given path
    def payoff(self, path, prevex, adjlogs=False):
        s = path[-1]
        if adjlogs: return np.log(s/self.strike)


        if self.optvar == CALL: return np.maximum(s - self.strike, 0) 
        if self.optvar == PUT:  return np.maximum(self.strike - s, 0) 

    # determining payoff coefficients
    def H(self, k):
        if self.optvar is CALL: return (self.chi(k, 0, self.b) - self.psi(k, 0, self.b)) * (2*self.strike)/(self.b-self.a)
        if self.optvar is PUT:  return (self.psi(k, self.a, 0) - self.chi(k, self.a, 0)) * (2*self.strike)/(self.b-self.a)

    # funtion returning characteristic function for maximum process starting at a time t
    def phi_adjlog_from(self, u, x, t):
        return self.process.phi_adjlog(u, x, t, self.T, self.strike)
    
    # funtion generating analytic solution
    def analytic(self, x, t):
        tau = self.T - t
        d1 = (np.log(x/self.strike) + (self.mu+self.sigma**2/2)*tau) / (self.sigma*math.sqrt(tau))
        d2 = d1 - self.sigma*math.sqrt(tau)

        if self.optvar is CALL: return x * norm.cdf(d1) - self.strike * np.exp(-self.mu*tau) * norm.cdf(d2)
        if self.optvar is PUT:  return self.strike * np.exp(-self.mu*tau) * norm.cdf(-d2) - x * norm.cdf(-d1)
    
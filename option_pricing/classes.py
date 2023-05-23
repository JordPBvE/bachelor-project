import numpy as np
import math
from enum import Enum
from scipy.stats import norm

# GENERAL CONSTANTS AND GENERAL FUNCTIONS
i = complex(0.0, 1.0)

class OptionVariant(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
PUT = OptionVariant.PUT
CALL = OptionVariant.CALL

def sump(array):
    if array.size == 0:
        return 0
    res = np.sum(array) - array[0]/2
    return res

def phi_N(u):
    return np.exp(-0.5 * u**2)

# CLASS DEFINING A GEOMETRIC BROWNIAN MOTION AND CHARACTERISTIC FUNCTIONS FOR PROCESSES DERIVED FROM GEOMETRIC BROWNIAN MOTIONS
class GBM:
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma
    
    def phi_adjlog(self, u, x, t, T, K):
        tau = T - t

        e1 = i*u*np.log(x/K)
        e2 = i*u*(self.mu - 0.5*self.sigma**2)*tau
        e3 = -0.5*tau*(self.sigma*u)**2

        return np.exp(e1 + e2 + e3)

    def phi_adjlogmax(self, u, x, t, T, K):
        tau = T - t
        alpha = (self.mu + 0.5*self.sigma**2) / self.sigma

        f1 = 2 * np.exp(i*u*np.log(x/K))
        f2 = 1 - alpha / (i*self.sigma*u + 2*alpha)
        f3 = np.exp(-i * (self.sigma*u - 2*alpha*i) * tau * alpha)
        f4 = phi_N(math.sqrt(tau) * (self.sigma * u - 2*i*alpha))

        return f1 * f2 * f3 * f4
    
    def phi_adjlogmin(self, u, x, t, T, K):
        tau = T - t
        alpha = (self.mu + (self.sigma**2)/2) / self.sigma

        f1 = 2 * np.exp(i*u*np.log(x/K))
        f2 = 1 - alpha / (i*self.sigma*u + 2*alpha)
        f3 = np.exp(i * (2*alpha*i - self.sigma*u) * tau * alpha)
        f4 = phi_N(math.sqrt(tau) * (2*i*alpha - self.sigma*u))

        return f1 * f2 * f3 * f4

# PARENT CLASS DEFINING OPTIONS 
class Option:
    def __init__(self, gbmprocess, exercisetime, strikeprice, optionvariant, lowerintegration, upperintegration):
        self.gbm    = gbmprocess
        self.mu     = gbmprocess.mu
        self.sigma  = gbmprocess.sigma
        self.T      = exercisetime
        self.strike = strikeprice
        self.optvar = optionvariant
        self.a      = lowerintegration
        self.b      = upperintegration

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
        
# CHILD OPTION CLASS DESCRIBING COS METHOD PARAMETERS FOR LOOKBACK OPTIONS
class Lookback(Option):
    def H(self, k, prevex):
        adjprev = np.log(prevex/self.strike)

        tc1 = tc2 = tc3 = 0
        tp1 = tp2 = tp3 = 0
        
        if prevex > self.strike:
            tc1 = self.psi(k, 0, adjprev) * 2*(prevex-self.strike)/(self.b-self.a)
            tc2 = (self.chi(k, adjprev, self.b) - self.psi(k, adjprev, self.b)) * 2*self.strike/(self.b - self.a)
            tp3 = (self.psi(k, self.a, 0) - self.chi(k, self.a, 0))* 2*self.strike/(self.b - self.a)
        else:
            tc3 = (self.chi(k, 0, self.b) - self.psi(k, 0, self.b))* 2*self.strike/(self.b - self.a)
            tp1 = self.psi(k, adjprev, 0) * 2*(self.strike-prevex)/(self.b-self.a)
            tp2 = (self.psi(k, self.a, adjprev) - self.chi(k, self.a, adjprev)) * 2*self.strike/(self.b - self.a)

        if self.optvar is CALL: return tc1 + tc2 + tc3
        if self.optvar is PUT:  return tp1 + tp2 + tp3

        
    def phi_adjlog_from(self, u, x, t):
        if self.optvar is CALL: return self.gbm.phi_adjlogmax(u, x, t, self.T, self.strike)
        if self.optvar is PUT:  return self.gbm.phi_adjlogmin(u, x, t, self.T, self.strike)

    def d(self, sub, x, t):
        tau = self.T - t
        return (np.log(x/sub) + self.mu*tau + self.sigma**2*tau/2) / (self.sigma*math.sqrt(tau))
    
    def analytic(self, x, t, prevex):
        tau = self.T - t

        tc1 = lambda input : x * norm.cdf(self.d(input, x, t) - np.exp(-self.mu*tau) * input * norm.cdf(self.d(input, x, t) - self.sigma*math.sqrt(tau)))
        tc2 = lambda input : np.exp(-self.mu*tau)*(self.sigma**2/(2*self.mu)) * x * (-(x/input)**(-(2*self.mu)/(self.sigma**2)) * norm.cdf(self.d(input, x, t) - 2*self.mu*math.sqrt(tau)/self.sigma) + np.exp(self.mu*tau)*norm.cdf(self.d(input, x, t)))

        tp1 = lambda input : -x * norm.cdf(-self.d(input, x, t) + np.exp(-self.mu*tau) * input * norm.cdf(self.sigma*math.sqrt(tau) - self.d(input, x, t)))
        tp2 = lambda input : np.exp(-self.mu*tau)*(self.sigma**2/(2*self.mu)) * x * ((x/input)**(-(2*self.mu)/(self.sigma**2)) * norm.cdf(2*self.mu*math.sqrt(tau)/self.sigma - self.d(input, x, t)) - np.exp(self.mu*tau)*norm.cdf(-self.d(input, x, t)))


        if self.optvar == CALL and prevex <  self.strike: return tc1(self.strike) + tc2(self.strike) 
        if self.optvar == CALL and prevex >= self.strike: return tc1(prevex) + tc2(prevex) + np.exp(-self.mu*tau)*(prevex-self.strike)

        if self.optvar == PUT  and prevex >= self.strike: return tp1(self.strike) + tp2(self.strike)
        if self.optvar == PUT  and prevex <  self.strike: return tc1(prevex) + tc2(prevex) + np.exp(-self.mu*tau)*(self.strike-prevex)


# CHILD OPTION CLASS DESCRIBING COS METHOD PARAMETERS FOR RUROPEAN OPTIONS
class European(Option):
    def H(self, k):
        if self.optvar is CALL: return (self.chi(k, 0, self.b) - self.psi(k, 0, self.b)) * (2*self.strike)/(self.b-self.a)
        if self.optvar is PUT:  return (self.psi(k, self.a, 0) - self.chi(k, self.a, 0)) * (2*self.strike)/(self.b-self.a)

    def phi_adjlog_from(self, u, x, t):
        return self.gbm.phi_adjlog(u, x, t, self.T, self.strike)
    
    def phi_test(self, u, x, t):
        tau = self.T - t
        return np.exp(i*u*np.log(x/self.strike)) * np.exp((self.mu - 0.5 * np.power(self.sigma, 2.0)) * i * u * tau - 0.5 * np.power(self.sigma, 2.0) * np.power(u, 2.0) * tau)
    
    def analytic(self, x, t):
        tau = self.T - t
        d1 = (np.log(x/self.strike) + (self.mu+self.sigma**2/2)*tau) / (self.sigma*math.sqrt(tau))
        d2 = d1 - self.sigma*math.sqrt(tau)

        if self.optvar is CALL: return x * norm.cdf(d1) - self.strike * np.exp(-self.mu*tau) * norm.cdf(d2)
        if self.optvar is PUT:  return self.strike * np.exp(-self.mu*tau) * norm.cdf(-d2) - x * norm.cdf(-d1)
    
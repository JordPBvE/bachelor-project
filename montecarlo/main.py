import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st


def main():
    figure, axis = plt.subplots(2)
    figure.set_size_inches(7, 10)
    axis[0].grid()
    axis[0].set(xlabel='t', ylabel='S(t)')
    axis[1].grid()
    axis[1].set(xlabel='V(T)', ylabel='density')

    N  = 1000
    T  = 2
    n  = 1000
    s0 = 1
    alpha = 0
    sigma = 0.20

    generatepaths(axis[0], axis[1], N, T, n, s0, alpha, sigma)

    plt.savefig('./montecarlo/fig.png')



def generatepaths(pathfig, endsfig, N, T, n, s0, alpha, sigma):
    vs = np.zeros(N)

    ts = np.arange(n * T) / n
    for i in range(N):
        gbm = generateGBM(T, n, s0, alpha, sigma)
        pathfig.plot(ts, gbm)

        vs[i] = np.max(gbm) - gbm[-1]
    
    endsfig.hist(vs, density=True, bins=40)


def generateGBM(T, n, s0, alpha, sigma):
    srw = (1 / np.sqrt(n)) * generateRW(n * T)
    ts  = np.arange(n * T) / n

    Sn = s0 * np.exp((alpha - np.power(sigma, 2)/2) * ts + sigma * srw)

    return Sn


def generateRW(n):
    Mn = np.zeros(n)

    for i in range(1, n):
        Mn[i] = Mn[i-1] + random.choice([-1, 1])

    return Mn

main()

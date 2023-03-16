import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def main():
    # setting integration boundaries
    a = -10.0
    b = 10.0

    # matplotlib setup
    figure, axis = plt.subplots(2)
    figure.set_size_inches(7, 10)
    axis[0].grid()
    axis[0].set(xlabel='x', ylabel='$f_X(x)$')
    axis[1].grid()
    axis[1].set(xlabel='y', ylabel='$f_Y(y)$')

    x = np.linspace(-5.0, 5.0, 100)
    y = np.linspace(0.05, 5.0, 1000)

    # defining the characteristic function for X
    mu_x = 0.0
    sigma_x = 1.0
    normal_phi_x = normalchar(mu_x, sigma_x)

    # density function for normal distribution: only for error checking
    normaldist = st.norm.pdf(x, mu_x, sigma_x)

    # plotting the resulting density function for multiple values of N
    for n in [2**k for k in range(2, 7)]:
        f_X = cosdensityfunction(normal_phi_x, n, a, b, x)
        error = np.max(np.abs(f_X - normaldist))
        print("For {0} expanansion terms the error is {1}".format(n, error))

        axis[0].plot(x, f_X)

    # defining the characteristic function for Y' => f_Y = 1/y * f_Y'(log(y))
    mu_y = 0.5
    sigma_y = 0.2
    normal_phi_y = normalchar(mu_y, sigma_y)

    for n in [16, 64, 128]:
        f_Y = 1/y * cosdensityfunction(normal_phi_y, n, a, b, np.log(y))
        axis[1].plot(y, f_Y)

    plt.savefig('./density_approximation/fig.png')

# function defining density function according to characteristic function


def cosdensityfunction(phi, N, a, b, x):
    i = complex(0.0, 1.0)
    ks = np.linspace(0, N-1, N)  # defining an array with all values of k
    us = ks * np.pi / (b - a)    # defining an array of the factor k*pi/(b-a)

    # defining the array of F_k values
    F_ks = (2.0 / (b - a)) * np.real(phi(us) * np.exp(-i * us * a))
    F_ks[0] = F_ks[0] * 0.5

    # defining array of cos(u*(x-a))
    cs = np.cos(np.outer(us, x - a))

    # inproduct of F_ks and cs => desired function
    f_X = np.matmul(F_ks, cs)

    return f_X

# function generating characteristic function for normal distribution


def normalchar(mu, sigma):
    i = complex(0.0, 1.0)
    return (lambda x: np.exp(i * x * mu - 0.5 * np.power(sigma, 2) * np.power(x, 2)))


main()

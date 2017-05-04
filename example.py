
from __future__ import  division
import numpy as np
from matplotlib import pyplot as plt
import os

from ars import ARS

_save_path = os.path.dirname(__file__) + '/result'

######################################
# Example 1: sample 10000 values
# from a Beta(1.5,1) distribution
######################################
def f(x, a=1.5, b=1):
    """
    Log beta distribution
    """
    return (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)


def fprima(x, a=1.5, b=1):
    """
    Derivative of Log beta distribution
    """
    return (a - 1) / x - (b - 1) / (1 - x)


# x = np.linspace(-100,100,100)
a = 1.5
b = 1
ars = ARS(f, fprima, xi=[0.1, 0.6], lb=0, ub=1, a=a, b=b)
samples = ars.draw(10000)

## mean = a/(a+b)
print np.mean(samples)

plt.hist(samples, bins=100, normed=True)
plt.savefig(_save_path + '/Beta(1.5,1).png')
plt.savefig(_save_path + '/Beta(1.5,1).pdf')
plt.show()


######################################
# Example 2: sample 10000 values
# from the normal distribution N(0,1)
######################################
def f(x, mu=0, sigma=1):
    """ 
    Log Normal distribution 
    """
    return -1/(2*sigma**2)*(x-mu)**2
    
def fprima(x, mu=0, sigma=1):
    """
    Derivative of Log Normal distribution
    """
    return -1/sigma**2*(x-mu)

# x = np.linspace(-100,100,100)
mu = 0
sigma = 1
ars = ARS(f, fprima, xi = [-4,1,400], mu=mu, sigma=sigma)


samples = ars.draw(10000)
## mean = mu
print np.mean(samples)

plt.hist(samples, bins=100, normed=True)
plt.savefig(_save_path + '/normal(0,1).png')
plt.savefig(_save_path + '/normal(0,1).pdf')
plt.show()


######################################
# Example 3: sample 10000 values
# from a Gamma(shape=2,scale=0.5)
######################################
def f(x, shape, scale=1):
    """ 
    Log gamma distribution 
    """
    return (shape-1)*np.log(x)-x/scale
    
    
def fprima(x, shape, scale=1):
    """
    Derivative of Log gamma distribution
    """
    return (shape-1)/x-1/scale

# x = np.linspace(-100,100,100)
ars = ARS(f, fprima, xi = [0.1,1,40], lb=0, shape=2, scale=0.5)
samples = ars.draw(10000)
## mean = shape*scale
print np.mean(samples)

plt.hist(samples, bins=100, normed=True)
plt.savefig(_save_path + '/Gamma(shape=2,scale=0.5).png')
plt.savefig(_save_path + '/Gamma(shape=2,scale=0.5).pdf')
plt.show()



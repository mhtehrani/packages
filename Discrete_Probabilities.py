"""
Descrete Probability Distributions:
    Binomial
    Hypergeometric
    Poisson
    Negative Binomial
    Descrete Uniform
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Binomial Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    """
    Binomial distribution describes the probability of x successes in n draws with replacement
    """
    
    # parameters
    n = 5 # number of trials
    p = 0.4 # probability of sucess
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.binom.stats(n, p, moments='mvsk')
    
    # Probability mass function - P(Y=x)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.arange(stats.binom.ppf(0.01, n, p), stats.binom.ppf(0.99, n, p)) # ppf(0.99, n, p) is Percent point function (inverse of cdf — percentiles)
    # plotting blue markers
    ax.plot(x, stats.binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf') # pmf(x, n, p) is Probability mass function
    # plotting vertical lines
    ax.vlines(x, 0, stats.binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
    plt.show()
    
    # Cumulative distribution function F(Y<=x)
    F_x = stats.binom.cdf(1, n, p, loc=0) # cdf(k, n, p, loc=0) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.binom.rvs(n, p, size=1000)




# Hypergeometric  Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
    """
    Hypergeometric distribution describes the probability of "x" successes (random 
    draws for which the object drawn has a specified feature) in "N" draws, without 
    replacement, from a finite population of size "M" that contains exactly "n" objects 
    with that feature, wherein each draw is either a success or a failure.
    """
    
    # parameters
    M = 20 # total number of elements in the population (population size)
    n = 10 # number of sucess in the population
    N = 11 # number of trials or draws
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.hypergeom.stats(M, n, N, moments='mvsk')
    
    # Probability mass function - P(Y=x)
    fig, ax = plt.subplots(1, 1)
    # The range of values 
    x = np.arange(0, n+1)
    # plotting blue markers
    ax.plot(x, stats.hypergeom.pmf(x, M, n, N), 'bo', ms=8, label='Hypergeometric pmf') # pmf(x, M, n, N) is Probability mass function
    # plotting vertical lines
    ax.vlines(x, 0, stats.hypergeom.pmf(x, M, n, N), colors='b', lw=5, alpha=0.5)
    plt.show()
    
    # Cumulative distribution function F(Y<=x)
    F_x = stats.hypergeom.cdf(3, M, n, N) # cdf(k, M, n, N) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.hypergeom.rvs(M, n, N, size=1000)




# Poisson Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
    """
    Poisson distribution describes a sequence of i.i.d. Bernoulli trials, of a given 
    number of events occurring in a fixed interval of time or space if these events 
    occur with a known constant mean rate "lambda"
    
    k is the number of occurrences
    lambda is mean number of occurrences (mu in scipy website)
    
    #####examples
    A call center receives an average of 180 calls per hour, 24 hours a day. The 
    calls are independent; receiving one does not change the probability of when 
    the next one will arrive. The number of calls received during any minute has 
    a Poisson probability distribution with mean 3: the most likely numbers are 2 
    and 3 but 1 and 4 are also likely and there is a small probability of it being 
    as low as zero and a very small probability it could be 10.
    
    Number of decay events that occur from a radioactive source during a defined 
    observation period.
    """
    
    # parameters
    lambda_ = 0.6 # mean number of occurrences (mu in scipy website)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.poisson.stats(lambda_, moments='mvsk')
    
    # Probability mass function - P(Y=x)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.arange(stats.poisson.ppf(0.01, lambda_), stats.poisson.ppf(0.99, lambda_)) # ppf(q, lambda_) is Percent point function (inverse of cdf — percentiles)
    # plotting blue markers
    ax.plot(x, stats.poisson.pmf(x, lambda_), 'bo', ms=8, label='Poisson pmf') # pmf(x, lambda_) is Probability mass function
    # plotting vertical lines
    ax.vlines(x, 0, stats.poisson.pmf(x, lambda_), colors='b', lw=5, alpha=0.5)
    plt.show()
    
    # Cumulative distribution function F(Y<=x)
    F_x = stats.poisson.cdf(2, lambda_) # cdf(k, lambda_) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.poisson.rvs(lambda_, size=1000)




# Negative Binomial
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html#scipy.stats.nbinom
    """
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli trials, 
    repeated until a predefined, (non-random) number of successes "n" occurs.
    
    n is the number of successes
    k is the number of failures
    N is total number of trials (n+k)
    p is the probability of a single success
    q is the probability of a single failure
    
    Another common parameterization of the negative binomial distribution is in terms 
    of the mean number of failures mu to achieve n successes.
    
    #####examples
    We can define rolling a 6 on a die as a success, and rolling any other number 
    as a failure, and ask how many failure rolls will occur before we see the third 
    success (r=3).
    
    We can use the negative binomial distribution to model the number of days n 
    (random) a certain machine works (specified by r) before it breaks down.
    
    We can model disease transmission for infectious diseases where the likely number 
    of onward infections may vary considerably from individual to individual and 
    from setting to setting.
    """
    
    # parameters
    n = 6 # number of successes
    p = 0.6 # probability of a single success
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.nbinom.stats(n, p, moments='mvsk')
    
    # Probability mass function - P(Y=x)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.arange(stats.nbinom.ppf(0.01, n, p), stats.nbinom.ppf(0.99, n, p)) # ppf(q, n, p) is Percent point function (inverse of cdf — percentiles)
    # plotting blue markers
    ax.plot(x, stats.nbinom.pmf(x, n, p), 'bo', ms=8, label='Poisson pmf') # pmf(x, n, p) is Probability mass function
    # plotting vertical lines
    ax.vlines(x, 0, stats.nbinom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
    plt.show()
    
    # Cumulative distribution function F(Y<=x)
    F_x = stats.nbinom.cdf(2, n, p) # cdf(k, n, p) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.nbinom.rvs(n, p, size=1000)




# Descrete Uniform Distribution (Random Integer)
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/tutorial/stats/discrete_randint.html
    
    # parameters
    low = 7 # low
    high = 31 # high
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.randint.stats(low, high, moments='mvsk')
    
    # Probability mass function - P(Y=x)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.arange(stats.randint.ppf(0.01, low, high), stats.randint.ppf(0.99, low, high)) # ppf(q, low, high) is Percent point function (inverse of cdf — percentiles)
    # plotting blue markers
    ax.plot(x, stats.randint.pmf(x, low, high), 'bo', ms=8, label='Poisson pmf') # pmf(x, low, high) is Probability mass function
    # plotting vertical lines
    ax.vlines(x, 0, stats.randint.pmf(x, low, high), colors='b', lw=5, alpha=0.5)
    plt.show()
    
    # Cumulative distribution function F(Y<=x)
    F_x = stats.randint.cdf(18, low, high) # cdf(k, low, high) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.randint.rvs(low, high, size=1000)


















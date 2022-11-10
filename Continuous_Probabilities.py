"""
Continuous Probability Distributions:
    Normal (Gaussian) Distribution
    Exponential  Distribution
    Continuous Uniform Distribution
    Weibull Distribution
    Gamma Distribution
    Triangular Distribution
    Chi-squared Distribution
    F Distribution
    Student’s t Distribution
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



# Normal (Gaussian) Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    
    # parameters
    mean = 1 # mean (loc in the scipy model)
    sd = 1 # standard deviation (scale in the scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.norm.stats(loc=mean, scale=sd, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.linspace(stats.norm.ppf(0.01, loc=mean, scale=sd), stats.norm.ppf(0.99, loc=mean, scale=sd), 100) # ppf(0.99, loc=mean, scale=sd) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.norm.pdf(x, loc=mean, scale=sd), 'r-', lw=3, alpha=0.6, label='norm pdf') # pdf(x, loc=mean, scale=sd) is Probability density function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.norm.cdf(0, loc=mean, scale=sd) # cdf(x, loc=mean, scale=sd) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.norm.rvs(loc=mean, scale=sd, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.norm.pdf(x, loc=mean, scale=sd), 'r-', lw=3, alpha=0.6, label='norm pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Exponential  Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    
    # parameters
    beta = 1./2 # one over the rate of occurance (mean of the exponentially distributed data) (1/lambda in exponential distribution | scale in scipy model)
    initial = 0 # the initial point of time or interval (usually the distribution starts from 0, since this is a memoryless distribution, this parameters ONLY shift the data and doesn't affect the distribution)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.expon.stats(loc=initial, scale=beta, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.expon.ppf(0, loc=initial, scale=beta), stats.expon.ppf(0.99, loc=initial, scale=beta), 100) # ppf(0.99, loc=initial, scale=beta) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.expon.pdf(x, loc=initial, scale=beta), 'r-', lw=3, alpha=0.6, label='expon pdf') # pdf(x, loc=initial, scale=beta) is Probability density function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.expon.cdf(3, loc=initial, scale=beta) # cdf(k, loc=initial, scale=beta) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.expon.rvs(loc=initial, scale=beta, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.expon.pdf(x, loc=initial, scale=beta), 'r-', lw=3, alpha=0.6, label='expon pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Continuous Uniform Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    
    # parameters
    low = 7 # low (loc in scipy model)
    dist = 31-7 # the ditance from the low point (high=low+dist) (scale in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.uniform.stats(loc=low, scale=dist, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0.01 and 0.99
    x = np.linspace(stats.uniform.ppf(0.01, loc=low, scale=dist), stats.uniform.ppf(0.99, loc=low, scale=dist), 100) # ppf(q, loc=low, scale=dist) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.uniform.pdf(x, loc=low, scale=dist), 'r-', lw=3, alpha=0.6, label='uniform pdf') # pmf(x, low, high) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.uniform.cdf(19, loc=low, scale=dist) # cdf(k, loc=low, scale=dist) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.uniform.rvs(loc=low, scale=dist, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.uniform.pdf(x, loc=low, scale=dist), 'r-', lw=3, alpha=0.6, label='uniform pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Weibull Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
    
    # parameters
    beta = 1.5 # shape parameter (c in scipy model)
    theta = 1 # scale parameter (scale in scipy model)
    shift = 0 # to shift the distribution (loc in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.weibull_min.stats(beta, loc=shift, scale=theta, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.weibull_min.ppf(0, beta, loc=shift, scale=theta), stats.weibull_min.ppf(0.99, beta, loc=shift, scale=theta), 100) # ppf(q, beta, loc=shift, scale=theta) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.weibull_min.pdf(x, beta, loc=shift, scale=theta), 'r-', lw=5, alpha=0.6, label='weibull_min pdf') # pmf(x, beta, loc=shift, scale=theta) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.weibull_min.cdf(2, beta, loc=shift, scale=theta) # cdf(k, beta, loc=shift, scale=theta) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.weibull_min.rvs(beta, loc=shift, scale=theta, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.weibull_min.pdf(x, beta, loc=shift, scale=theta), 'r-', lw=5, alpha=0.6, label='weibull_min pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Gamma Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """
    
    *****Examples
    We can use to model waiting times. For instance, in life testing, the waiting 
    time until death.
    """
    
    # parameters
    alpha = 2 # shape parameter (a in scipy model)
    beta = 2 # scale parameter (scale in scipy model) if rate is given --> beta=1/rate
    shift = 0 # to shift the distribution (loc in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.gamma.stats(alpha, loc=shift, scale=beta, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.gamma.ppf(0, alpha, loc=shift, scale=beta), stats.gamma.ppf(0.99, alpha, loc=shift, scale=beta), 100) # ppf(q, alpha, loc=shift, scale=beta) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.gamma.pdf(x, alpha, loc=shift, scale=beta), 'r-', lw=3, alpha=0.6, label='gamma pdf') # pmf(x, alpha, loc=shift, scale=beta) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.gamma.cdf(2, alpha, loc=shift, scale=beta) # cdf(k, alpha, loc=shift, scale=beta) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.gamma.rvs(alpha, loc=shift, scale=beta, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.gamma.pdf(x, alpha, loc=shift, scale=beta), 'r-', lw=3, alpha=0.6, label='gamma pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Triangular Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html
    
    # parameters
    loc = 0 # the lower bound of the distribution
    c = 0.8 # shape parameter (the ratio of the mode to scale)
    dist = 10 # the distance from the lower bound to the upper bound
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.triang.stats(c, loc=loc, scale=dist, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 1
    x = np.linspace(stats.triang.ppf(0, c, loc=loc, scale=dist), stats.triang.ppf(1, c, loc=loc, scale=dist), 100) # ppf(q, c, loc=loc, scale=dist) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.triang.pdf(x, c, loc=loc, scale=dist), 'r-', lw=3, alpha=0.6, label='triang pdf') # pmf(x, c, loc=loc, scale=dist) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.triang.cdf(2, c, loc=loc, scale=dist) # cdf(k, c, loc=loc, scale=dist) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.triang.rvs(c, loc=loc, scale=dist, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.triang.pdf(x, c, loc=loc, scale=dist), 'r-', lw=3, alpha=0.6, label='triang pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Chi-squared Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    
    # parameters
    df = 5 # degree of freedom
    shift = 0 # to shift the distribution (loc in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.chi2.stats(df, loc=shift, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.chi2.ppf(0, df, loc=shift), stats.chi2.ppf(0.99, df, loc=shift), 100) # ppf(q, df, loc=shift) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.chi2.pdf(x, df, loc=shift), 'r-', lw=3, alpha=0.6, label='chi2 pdf') # pmf(x, df, loc=shift) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.chi2.cdf(2, df, loc=shift) # cdf(k, df, loc=shift) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.chi2.rvs(df, loc=shift, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.chi2.pdf(x, df, loc=shift), 'r-', lw=3, alpha=0.6, label='chi2 pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# F Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
    
    # parameters
    df1 = 29 # degree of freedom 1
    df2 = 18 # degree of freedom 2
    shift = 0 # to shift the distribution (loc in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.f.stats(df1, df2, loc=shift, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.f.ppf(0, df1, df2, loc=shift), stats.f.ppf(0.99, df1, df2, loc=shift), 100) # ppf(q, df1, df2, loc=shift) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.f.pdf(x, df1, df2, loc=shift), 'r-', lw=3, alpha=0.6, label='f pdf') # pmf(x, df1, df2, loc=shift) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.f.cdf(2, df1, df2, loc=shift) # cdf(k, df1, df2, loc=shift) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.f.rvs(df1, df2, loc=shift, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.f.pdf(x, df1, df2, loc=shift), 'r-', lw=3, alpha=0.6, label='f pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()




# Student’s t Distribution
#==============================================================================
if True:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    
    # parameters
    df = 71 # degree of freedom
    shift = 0 # to shift the distribution (loc in scipy model)
    
    # Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
    mean, var, skew, kurt = stats.t.stats(df, loc=shift, moments='mvsk')
    
    # Probability density function - P(y)
    fig, ax = plt.subplots(1, 1)
    # The range of values that cover from 0 and 0.99
    x = np.linspace(stats.t.ppf(0.01, df, loc=shift), stats.t.ppf(0.99, df, loc=shift), 100) # ppf(q, df, loc=shift) is Percent point function (inverse of cdf — percentiles)
    # plotting the distribution
    ax.plot(x, stats.t.pdf(x, df, loc=shift), 'r-', lw=3, alpha=0.6, label='t pdf') # pmf(x, df, loc=shift) is Probability mass function
    plt.show()
    
    # Cumulative distribution function F(Y<=k)
    F_x = stats.t.cdf(2, df, loc=shift) # cdf(k, df, loc=shift) is Cumulative distribution function
    
    # Generate random numbers
    r = stats.t.rvs(df, loc=shift, size=1000)
    
    # pdf on top of histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.plot(x, stats.t.pdf(x, df, loc=shift), 'r-', lw=3, alpha=0.6, label='t pdf')
    ax.legend(loc='best', frameon=False)
    plt.show()































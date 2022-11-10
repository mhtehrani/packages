"""
Statistical Test:
"""

# Importing required libraries
#==============================================================================
import matplotlib.pyplot as plt
import numpy as np



#==============================================================================
#==============================================================================
# Statistical Test for population mean (mu) - Sigma known
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the population mean is equal to, smaller than, or greater than the hypothesized value

Hypothesis:
    H0:	The population mean (mu) is equal to y_bar (sample mean)
    Ha:	The population mean (mu) is not equal to y_bar (sample mean)
    
    Somehow (unrealisticly) we know that the population standard deviation is Sigma
    
Assumptions:
    We know that the population distribution is normal or we have enough sample (n>=30) so that central limit theorim kicks in

Test Statistic:
    Z0 = (y_bar-mu)/(Sigma/sqrt(n))
    
Distribution:
    Z (standard normal) - two-tailed test

E.g.: It is claimed that the population mean is equal to 230, we know the population standard deviation is 60
'''
n = 20
Sigma = 60
mu = 230

# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=80, size=n)

# sample mean
y_bar = np.mean(sample)
# test_statistics
test_statistics_Z0 = (y_bar-mu)/(Sigma/np.sqrt(n))
# p_value on the two tail test
p_value = 2*norm.cdf(test_statistics_Z0, loc=0, scale=1)


'''
Hypothesis:
    H0:	The population mean (mu) is equal to or greater than y_bar (sample mean)
    Ha:	The population mean (mu) is smaller than y_bar (sample mean)

Distribution:
    Z (standard normal) - one-tailed test (left-tail)

E.g.: It is claimed that the population mean is equal to or greater than 230, we know the population standard deviation is 60
'''
n = 20
Sigma = 60
mu = 230

# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=80, size=n)

# sample mean
y_bar = np.mean(sample)
# test_statistics
test_statistics_Z0 = (y_bar-mu)/(Sigma/np.sqrt(n))
# p_value on the one tail test
p_value = norm.cdf(test_statistics_Z0, loc=0, scale=1)


'''
Hypothesis:
    H0:	The population mean (mu) is equal to or smaller than y_bar (sample mean)
    Ha:	The population mean (mu) is greater than y_bar (sample mean)

Distribution:
    Z (standard normal) - one-tailed test (right-tail)

It is claimed that the population mean is qual to or smaller than 230, we know the population standard deviation is 60
'''
n = 20
Sigma = 60
mu = 230

# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=260, scale=80, size=n)

# sample mean
y_bar = np.mean(sample)
# test_statistics
test_statistics_Z0 = (y_bar-mu)/(Sigma/np.sqrt(n))
# p_value on the one tail test
p_value = 1 - norm.cdf(test_statistics_Z0, loc=0, scale=1)



#==============================================================================
#==============================================================================
# Statistical Test for population mean (mu) - Sigma unknown
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the population mean is equal to, smaller than, or greater than the hypothesized value

Hypothesis:
    H0:	The population mean (mu) is equal to y_bar (sample mean)
    Ha:	The population mean (mu) is not equal to y_bar (sample mean)

Assumptions:
    We know that the population distribution is normal or we have enough sample (n>=30) so that central limit theorim kicks in

Test Statistic:
    t0 = (y_bar-mu)/(sd/sqrt(n))
    
Distribution:
    t - two-tailed test

E.g.: It is claimed that the population mean is equal to 230
'''
n = 20
df = n-1
mu = 230
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=80, size=n)
# sample mean and standard deviation
y_bar = np.mean(sample)
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_t0 = (y_bar-mu)/(sd/np.sqrt(n))
# p_value on the two tail test
from scipy.stats import t
p_value = 2*t.cdf(test_statistics_t0, df, loc=0)


'''
Hypothesis:
    H0:	The population mean (mu) is equal to or greater than y_bar (sample mean)
    Ha:	The population mean (mu) is smaller than y_bar (sample mean)

Distribution:
    t - one-tailed test (left-tail)

E.g.: It is claimed that the population mean is equal to or greater than 230
'''
n = 20
df = n-1
mu = 230
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=80, size=n)
# sample mean and standard deviation
y_bar = np.mean(sample)
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_t0 = (y_bar-mu)/(sd/np.sqrt(n))
# p_value on the two tail test
from scipy.stats import t
p_value = t.cdf(test_statistics_t0, df, loc=0)


'''
Hypothesis:
    H0:	The population mean (mu) is equal to or smaller than y_bar (sample mean)
    Ha:	The population mean (mu) is greater than y_bar (sample mean)

Distribution:
    t - one-tailed test (right-tail)

E.g.: It is claimed that the population mean is equal to or smaller than 230
'''
n = 20
df = n-1
mu = 230
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=260, scale=80, size=n)
# sample mean and standard deviation
y_bar = np.mean(sample)
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_t0 = (y_bar-mu)/(sd/np.sqrt(n))
# p_value on the two tail test
from scipy.stats import t
p_value = 1-t.cdf(test_statistics_t0, df, loc=0)



#==============================================================================
#==============================================================================
# Statistical Test for population proportion (p)
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the population proportion is equal to, smaller than, or greater than the hypothesized value
The population can be divided into to subset of successes and failures

Hypothesis:
    H0:	The population proportion (p_hat) is equal to p0 (hypothesized proportion)
    Ha:	The population proportion (p_hat) is not equal to p0 (hypothesized proportion)

Assumptions:
    We have enough sample (np>=4 and n(1-p)>=4) so that central limit theorim kicks in

Test Statistic:
    z0 = (p_hat-p0)/s
    s = sqrt(p0*(1-p0)/n) * note that p0 is the population proportion not the sampled proportion

Distribution:
    Z - two-tailed test

E.g.: It is claimed that the population proportion is equal to 0.32
'''
n = 300
y = 92
p0 = 0.32
# proportion of success
p_hat = y/n
# standard deviation
s = np.sqrt(p0*(1-p0)/n)
# test_statistics
test_statistics_z0 = (p_hat-p0)/s
# p_value on the two tail test
if test_statistics_z0 < 0 :
    p_value = 2*norm.cdf(test_statistics_z0, loc=0, scale=1)
else:
    p_value = 2*(1-norm.cdf(test_statistics_z0, loc=0, scale=1))
#end

'''
Hypothesis:
    H0:	The population proportion (p_hat) is equal to or greater than p0 (hypothesized proportion)
    Ha:	The population proportion (p_hat) is less than p0 (hypothesized proportion)

Distribution:
    Z - one-tailed test (lower tail)

E.g.: It is claimed that the population proportion is equal to or greater than 0.08
'''
n = 300
y = 12
p0 = 0.08
# proportion of success
p_hat = y/n
# standard deviation
s = np.sqrt(p0*(1-p0)/n)
# test_statistics
test_statistics_z0 = (p_hat-p0)/s
# p_value on the lower tail test
p_value = norm.cdf(test_statistics_z0, loc=0, scale=1)

'''
Hypothesis:
    H0:	The population proportion (p_hat) is equal to or less than p0 (hypothesized proportion)
    Ha:	The population proportion (p_hat) is greater than p0 (hypothesized proportion)

Distribution:
    Z - one-tailed test (upper tail)

E.g.: It is claimed that the population proportion is less than or equal to 0.8
'''
n = 64
y = 55
p0 = 0.8
# proportion of success
p_hat = y/n
# standard deviation
s = np.sqrt(p0*(1-p0)/n)
# test_statistics
test_statistics_z0 = (p_hat-p0)/s
# p_value on the upper tail test
p_value = 1-norm.cdf(test_statistics_z0, loc=0, scale=1)



#==============================================================================
#==============================================================================
# Statistical Test for population variance (Sigma^2)
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the population variance is equal to, smaller than, or greater than the hypothesized value

Hypothesis:
    H0:	The population variance (Sigma^2) is equal to Sigma_hat^2 (sample variance)
    Ha:	The population variance (Sigma^2) is not equal to Sigma_hat^2 (sample variance)

Test Statistic:
    Chi0^2 = ((n-1)*sd^2)/(Sigma_hat^2)
    
Distribution:
    Chi-squared - two-tailed test

E.g.: It is claimed that the population variance is equal to 6100
'''
n = 40
df = n-1
Sigma_hat_2 = 6100
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=60, size=n)
# sample standard deviation
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_Chi_0_2 = ((n-1)*sd**2)/Sigma_hat_2
# p_value on the two tail test
from scipy.stats import chi2
p_value = 2*min(chi2.cdf(test_statistics_Chi_0_2, df, loc=0), 1-chi2.cdf(test_statistics_Chi_0_2, df, loc=0))


'''
Hypothesis:
    H0:	The population variance (Sigma^2) is equal to or greater than Sigma_hat^2 (sample variance)
    Ha:	The population variance (Sigma^2) is smaller than Sigma_hat^2 (sample variance)

Distribution:
    Chi-squared - one-tailed test (left-tail)

E.g.: It is claimed that the population variance is equal to or greater than 6100
'''
n = 60
df = n-1
Sigma_hat_2 = 6100
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=60, size=n)
# sample standard deviation
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_Chi_0_2 = ((n-1)*sd**2)/Sigma_hat_2
# p_value on the two tail test
from scipy.stats import chi2
p_value = chi2.cdf(test_statistics_Chi_0_2, df, loc=0)


'''
Hypothesis:
    H0:	The population variance (Sigma^2) is equal to or smaller than Sigma_hat^2 (sample variance)
    Ha:	The population variance (Sigma^2) is greater than Sigma_hat^2 (sample variance)

Distribution:
    Chi-squared - one-tailed test (right-tail)

E.g.: It is claimed that the population variance is equal to or smaller than 6100
'''
n = 60
df = n-1
Sigma_hat_2 = 6100
# generating normal samples
from scipy.stats import norm
sample = norm.rvs(loc=200, scale=95, size=n)
# sample standard deviation
sd = np.std(sample, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# test_statistics
test_statistics_Chi_0_2 = ((n-1)*sd**2)/Sigma_hat_2
# p_value on the two tail test
from scipy.stats import chi2
p_value = 1-chi2.cdf(test_statistics_Chi_0_2, df, loc=0)



#==============================================================================
#==============================================================================
# Statistical Test for difference in population means (mu1-mu2) - Sigma known
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the difference in population means is equal to, smaller than, or greater than the hypothesized value.
The samples from each population should be independent

Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is not equal to (y_bar1 - y_bar2) (sample means)
    
    Somehow (unrealisticly) we know the population standard deviations are Sigma1 and Sigma2
    
Assumptions:
    We know that the population distributions are normal or we have enough sample (n1, n2 >= 30) so that central limit theorim kicks in

Test Statistic:
    Z0 = [(y_bar1-y_bar2)-(mu1-mu2)]/sqrt(Sigma1^2/n1 + Sigma2^2/n2)
    
Distribution:
    Z (standard normal) - two-tailed test

E.g.: It is claimed that the difference in population means is equal to -40, we know the population standard deviations are Sigma1=60 and Sigma2=46
'''
n1 = 20
n2 = 23
Sigma1 = 60
Sigma2 = 46
d0 = -40 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=200, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=260, scale=Sigma2, size=n2)
# sample mean
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
# test_statistics
test_statistics_Z0 = ((y_bar1-y_bar2)-d0)/np.sqrt(Sigma1**2/n1 + Sigma2**2/n2)
# p_value on the two tail test
p_value = 2*norm.cdf(test_statistics_Z0, loc=0, scale=1)


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or greater than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is less than (y_bar1 - y_bar2) (sample means)

Test Statistic:
    Z0 = [(y_bar1-y_bar2)-(mu1-mu2)]/sqrt(Sigma1^2/n1 + Sigma2^2/n2)
    
Distribution:
    Z (standard normal) - one-tailed test (left-tail)

E.g.: It is claimed that the difference in population means is equal to -40, we know the population standard deviations are Sigma1=60 and Sigma2=46
'''
n1 = 20
n2 = 23
Sigma1 = 60
Sigma2 = 46
d0 = -40 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=200, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=260, scale=Sigma2, size=n2)
# sample mean
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
# test_statistics
test_statistics_Z0 = ((y_bar1-y_bar2)-d0)/np.sqrt(Sigma1**2/n1 + Sigma2**2/n2)
# p_value on the two tail test
p_value = norm.cdf(test_statistics_Z0, loc=0, scale=1)


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or less than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is greater than (y_bar1 - y_bar2) (sample means)

Test Statistic:
    Z0 = [(y_bar1-y_bar2)-(mu1-mu2)]/sqrt(Sigma1^2/n1 + Sigma2^2/n2)
    
Distribution:
    Z (standard normal) - one-tailed test (right-tail)

E.g.: It is claimed that the difference in population means is equal to 40, we know the population standard deviations are Sigma1=60 and Sigma2=46
'''
n1 = 20
n2 = 23
Sigma1 = 60
Sigma2 = 46
d0 = 40 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=200, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=260, scale=Sigma2, size=n2)
# sample mean
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
# test_statistics
test_statistics_Z0 = ((y_bar1-y_bar2)-d0)/np.sqrt(Sigma1**2/n1 + Sigma2**2/n2)
# p_value on the two tail test
p_value = 1-norm.cdf(test_statistics_Z0, loc=0, scale=1)



#==============================================================================
#==============================================================================
# Statistical Test for difference in population means (mu1-mu2) - Sigma unknown but assumed "equal"
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the difference in population means is equal to, smaller than, or greater than the hypothesized value.
The samples from each population should be independent

Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is not equal to (y_bar1 - y_bar2) (sample means)
    
    Somehow (unrealisticly) we know the population standard deviations are Sigma1 and Sigma2 and Sigma1=Sigma2
    
Assumptions:
    We know that the population distributions are normal or we have enough sample (n1, n2 >= 30) so that central limit theorem kicks in

Test Statistic:
    t0 = [(y_bar1-y_bar2)-(mu1-mu2)]/pooled_sigma
    pooled_sigma = sqrt( [(n1-1)sd1^2 + (n2-1)sd2^2]/df ) * sqrt(1/n1 + 1/n2)
    df = n1+n2-2
    
Distribution:
    t - two-tailed test

E.g.: It is claimed that the difference in population means is equal to 0, we know the population standard deviations are equal
'''
n1 = 42
n2 = 36
df = n1+n2-2
Sigma1 = 2
Sigma2 = 2
d0 = 0 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=4.1, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=4.5, scale=Sigma2, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
sd1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
sd2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# pooled_std
pooled_sigma = np.sqrt( ((n1-1)*sd1**2 + (n2-1)*sd2**2)/df ) * np.sqrt(1/n1 + 1/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
p_value = 2*np.minimum(t.cdf(test_statistics_t0, df, loc=0), 1- t.cdf(test_statistics_t0, df, loc=0))


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or greater than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is less than (y_bar1 - y_bar2) (sample means)

Distribution:
    t - one-tail test (left-tail)

E.g.: It is claimed that the difference in population means is equal to 0, we know the population standard deviations are equal
'''
n1 = 42
n2 = 36
df = n1+n2-2
Sigma1 = 2
Sigma2 = 2
d0 = 0 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=4.1, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=4.5, scale=Sigma2, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
sd1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
sd2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# pooled_std
pooled_sigma = np.sqrt( ((n1-1)*sd1**2 + (n2-1)*sd2**2)/df ) * np.sqrt(1/n1 + 1/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
p_value = t.cdf(test_statistics_t0, df, loc=0)


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or less than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is greater than (y_bar1 - y_bar2) (sample means)

Distribution:
    t - one-tail test (right-tail)

E.g.: It is claimed that the difference in population means is equal to 0, we know the population standard deviations are equal
'''
n1 = 42
n2 = 36
df = n1+n2-2
Sigma1 = 2
Sigma2 = 2
d0 = 0 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=4.1, scale=Sigma1, size=n1)
sample2 = norm.rvs(loc=4.5, scale=Sigma2, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
sd1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
sd2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# pooled_std
pooled_sigma = np.sqrt( ((n1-1)*sd1**2 + (n2-1)*sd2**2)/df ) * np.sqrt(1/n1 + 1/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
p_value = 1-t.cdf(test_statistics_t0, df, loc=0)



#==============================================================================
#==============================================================================
# Statistical Test for difference in population means (mu1-mu2) - Sigma unknown but assumed "Unequal"
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the difference in population means is equal to, smaller than, or greater than the hypothesized value.
The samples from each population should be independent

Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is not equal to (y_bar1 - y_bar2) (sample means)

Assumptions:
    We know that the population distributions are normal or we have enough sample (n1, n2 >= 30) so that central limit theorem kicks in

Test Statistic:
    t0 = [(y_bar1-y_bar2)-(mu1-mu2)]/pooled_sigma
    pooled_sigma = sqrt(s1^2/n1 + s2^2/n2)
    df = [s1^2/n1 + s2^2/n2]^2 / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]
    * Note that we need to round DOWN the df to the nearest integer value.

Distribution:
    t - two-tailed test

E.g.: It is claimed that the difference in population means is equal to 0
'''
n1 = 30
n2 = 30
d0 = 0 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=1284.4, scale=369.4, size=n1)
sample2 = norm.rvs(loc=1548.3, scale=259.7, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
s1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
s2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# degree of freedom
df = np.floor( (s1**2/n1 + s2**2/n2)**2 / ( (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1) ) )
# pooled_std
pooled_sigma = np.sqrt(s1**2/n1 + s2**2/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
if test_statistics_t0 < 0:
    p_value = 2*t.cdf(test_statistics_t0, df, loc=0)
else:
    p_value = 2*(1-t.cdf(test_statistics_t0, df, loc=0))
#end


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or greater than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is smaller than (y_bar1 - y_bar2) (sample means)

Distribution:
    t - One-tailed test (left-tail)

E.g.: It is claimed that the difference in population means is equal to or greater than -50
'''
n1 = 30
n2 = 30
d0 = -50 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=1284.4, scale=369.4, size=n1)
sample2 = norm.rvs(loc=1548.3, scale=259.7, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
s1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
s2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# degree of freedom
df = np.floor( (s1**2/n1 + s2**2/n2)**2 / ( (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1) ) )
# pooled_std
pooled_sigma = np.sqrt(s1**2/n1 + s2**2/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
p_value = t.cdf(test_statistics_t0, df, loc=0)


'''
Hypothesis:
    H0:	The difference in population means (mu1-mu2) is equal to or smaller than (y_bar1 - y_bar2) (sample means)
    Ha:	The difference in population means (mu1-mu2) is greater than (y_bar1 - y_bar2) (sample means)

Distribution:
    t - One-tailed test (right-tail)

E.g.: It is claimed that the difference in population means is equal to or smaller than 0
'''
n1 = 47
n2 = 39
d0 = 0 # mu1-mu2

# generating normal samples
from scipy.stats import norm
sample1 = norm.rvs(loc=56.9, scale=13.2, size=n1)
sample2 = norm.rvs(loc=48.4, scale=10.4, size=n2)
# sample means and standard deviations
y_bar1 = np.mean(sample1)
y_bar2 = np.mean(sample2)
s1 = np.std(sample1, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
s2 = np.std(sample2, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# degree of freedom
df = np.floor( (s1**2/n1 + s2**2/n2)**2 / ( (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1) ) )
# pooled_std
pooled_sigma = np.sqrt(s1**2/n1 + s2**2/n2)
# test_statistics
test_statistics_t0 = ((y_bar1-y_bar2)-d0)/pooled_sigma
# p_value on the two tail test
from scipy.stats import t
p_value = 1-t.cdf(test_statistics_t0, df, loc=0)



#==============================================================================
#==============================================================================
# Statistical Test for mean paired difference between two sets of observations mu_d (before-after)
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the mean paired difference is equal to, smaller than, or greater than the hypothesized value.
There is no independent samples from each population, instead, the population is gone through a change (before and after) and we study the effect of the change
Another situation is that the same operation is performed on the same set of samples

Hypothesis:
    H0:	The mean paired difference (d_bar) is equal to (d0)
    Ha:	The mean paired difference (d_bar) is not equal to (d0)

Test Statistic:
    t0 = (d_bar-d0)/(s/sqrt(n))
    df = n-1

Distribution:
    t - two-tailed test

E.g.: It is claimed that the mean paired difference is equal to 0
'''
n = 8
d0 = 0 # before-after
# generating normal samples
d_1 = [185.4, 146.3, 174.4, 184.9, 240.0, 253.8, 238.8, 263.5]
d_2 = [180.4, 248.5, 185.5, 216.4, 269.3, 249.6, 282.0, 315.9]
d_paired_diff = np.array(d_1)-np.array(d_2)
# sample means and standard deviations
d_bar = np.mean(d_paired_diff)
s = np.std(d_paired_diff, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements 
# degree of freedom
df = n-1
# test_statistics
test_statistics_t0 = (d_bar-d0)/(s/np.sqrt(n))
# p_value on the two tail test
from scipy.stats import t
if test_statistics_t0 < 0:
    p_value = 2*t.cdf(test_statistics_t0, df, loc=0)
else:
    p_value = 2*(1-t.cdf(test_statistics_t0, df, loc=0))
#end


'''
Hypothesis:
    H0:	The mean paired difference (d_bar) is equal to or greater than (d0)
    Ha:	The mean paired difference (d_bar) is less than (d0)

Distribution:
    t - one-tailed test (left tail)

E.g.: It is claimed that the mean paired difference is equal to or greater than -5
'''
n = 8
d0 = -5 # before-after
# generating normal samples
d_1 = [185.4, 146.3, 174.4, 184.9, 240.0, 253.8, 238.8, 263.5]
d_2 = [180.4, 248.5, 185.5, 216.4, 269.3, 249.6, 282.0, 315.9]
d_paired_diff = np.array(d_1)-np.array(d_2)
# sample means and standard deviations
d_bar = np.mean(d_paired_diff)
s = np.std(d_paired_diff, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements 
# degree of freedom
df = n-1
# test_statistics
test_statistics_t0 = (d_bar-d0)/(s/np.sqrt(n))
# p_value on the two tail test
from scipy.stats import t
p_value = t.cdf(test_statistics_t0, df, loc=0)


'''
Hypothesis:
    H0:	The mean paired difference (d_bar) is equal to or less than (d0)
    Ha:	The mean paired difference (d_bar) is greater than (d0)

Distribution:
    t - one-tailed test (right tail)

E.g.: It is claimed that the mean paired difference is equal to or less than 0
'''
n = 15
d0 = 0 # before-after
# generating normal samples
d_1 = [48.4, 47.73, 51.3, 50.49, 47.06, 53.02, 48.96, 52.03, 51.09, 47.35, 50.15, 46.59, 52.03, 51.96, 49.15]
d_2 = [51.14, 46.48, 50.9, 49.82, 47.99, 53.2, 46.76, 54.44, 49.85, 47.45, 50.66, 47.92, 52.37, 52.9, 50.67]
d_paired_diff = np.array(d_1)-np.array(d_2)
# sample means and standard deviations
d_bar = np.mean(d_paired_diff)
s = np.std(d_paired_diff, ddof=1) #ddof = The divisor used in calculations is N - ddof, where N represents the number of elements
# degree of freedom
df = n-1
# test_statistics
test_statistics_t0 = (d_bar-d0)/(s/np.sqrt(n))
# p_value on the two tail test
p_value = 1-t.cdf(test_statistics_t0, df, loc=0)



#==============================================================================
#==============================================================================
# Statistical Test for differences in population proportions
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the differences in the population proportions is equal to, smaller than, or greater than the hypothesized value.
The samples are independent.

Hypothesis:
    H0:	The differences in population proportions is equal to (p1-p2)
    Ha:	The differences in population proportions is not equal to (p1-p2)

Test Statistic:
    z0 = [(p1_hat-p2_hat)-(p1-p2)]/s
    s = sqrt( p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2 )

Distribution:
    Z - two-tailed test

E.g.: It is claimed that the differences in the population proportions is equal to 0
'''
n1 = 217
y1 = 189
n2 = 246
y2 = 204
d0 = 0 #p1-p2
# proportion of success
p1_hat = y1/n1
p2_hat = y2/n2
# standard deviation
s = np.sqrt( p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2 )
# test_statistics
test_statistics_z0 = ((p1_hat-p2_hat)-d0)/s
# p_value on the two tail test
if test_statistics_z0 < 0:
    p_value = 2*norm.cdf(test_statistics_z0, loc=0, scale=1)
else:
    p_value = 2*(1-norm.cdf(test_statistics_z0, loc=0, scale=1))
#end

'''
Hypothesis:
    H0:	The differences in population proportions is equal to or greater than (p1-p2)
    Ha:	The differences in population proportions is less than (p1-p2)

Distribution:
    Z - one-tailed test (left tail)

E.g.: It is claimed that the differences in the population proportions is equal to or greater than 0
'''
n1 = 200
y1 = 8
n2 = 250
y2 = 13
d0 = 0 #p1-p2
# proportion of success
p1_hat = y1/n1
p2_hat = y2/n2
# standard deviation
s = np.sqrt( p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2 )
# test_statistics
test_statistics_z0 = ((p1_hat-p2_hat)-d0)/s
# p_value on the lower tail test
p_value = norm.cdf(test_statistics_z0, loc=0, scale=1)

'''
Hypothesis:
    H0:	The differences in population proportions is equal to or less than (p1-p2)
    Ha:	The differences in population proportions is greater than (p1-p2)

Distribution:
    Z - one-tailed test (left tail)

E.g.: It is claimed that the differences in the population proportions is equal to or less than 0
'''
n1 = 217
y1 = 189
n2 = 246
y2 = 204
d0 = 0 #p1-p2
# proportion of success
p1_hat = y1/n1
p2_hat = y2/n2
# standard deviation
s = np.sqrt( p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2 )
# test_statistics
test_statistics_z0 = ((p1_hat-p2_hat)-d0)/s
# p_value on the lower tail test
p_value = 1-norm.cdf(test_statistics_z0, loc=0, scale=1)



#==============================================================================
#==============================================================================
# Statistical Test for differences in population variances
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the differences in population variances is equal to, smaller than, or greater than 0.
The samples are independent.

Hypothesis:
    H0:	The differences in population variances is equal to 0
    Ha:	The differences in population variances is not equal to 0

Test Statistic:
    F0 = s1^2/s2^2

Distribution:
    F - two-tailed test

E.g.: It is claimed that the differences in population variances is equal to 0
'''
n1 = 27
s1 = 0.35
n2 = 32
s2 = 0.40
df1 = n1-1 # degree of freedom 1
df2 = n2-1 # degree of freedom 2
# test_statistics
test_statistics_F0 = s1**2/s2**2
# p_value on the two tail test
from scipy.stats import f
p_value = np.minimum(2*f.cdf(test_statistics_F0, df1, df2, loc=0), 2*(1-f.cdf(test_statistics_F0, df1, df2, loc=0)))

'''
Hypothesis:
    H0:	The differences in population variances is equal to or greater than 0
    Ha:	The differences in population variances is smaller than 0

Distribution:
    F - One-tailed test (left tail)

E.g.: It is claimed that the ratio of population variances is equal to or greater than 0
'''
n1 = 25
s1 = 2.3
n2 = 16
s2 = 1.1
d0 = 0 #Sigma1^2-Sigma2^2
df1 = n1-1 # degree of freedom 1
df2 = n2-1 # degree of freedom 2
# test_statistics
test_statistics_F0 = s1**2/s2**2
# p_value on the lower tail test
from scipy.stats import f
p_value = f.cdf(test_statistics_F0, df1, df2, loc=0)

'''
Hypothesis:
    H0:	The differences in population variances is equal to or smaller than 0
    Ha:	The differences in population variances is greater than 0

Distribution:
    F - One-tailed test (right tail)

E.g.: It is claimed that the ratio of population variances is equal to or smaller than 0
'''
n1 = 25
s1 = 2.3
n2 = 16
s2 = 1.1
d0 = 0 #Sigma1^2-Sigma2^2
df1 = n1-1 # degree of freedom 1
df2 = n2-1 # degree of freedom 2
# test_statistics
test_statistics_F0 = s1**2/s2**2
# p_value on the upper tail test
from scipy.stats import f
p_value = 1-f.cdf(test_statistics_F0, df1, df2, loc=0)



#==============================================================================
#==============================================================================
# Sign Test (Nonparametric) - Test for central tendency of a population that cannot be tested via parametric tests
#==============================================================================
#==============================================================================
'''
Purpose: can be used to see if the central tendency of a population is equal to, smaller than, or greater than the hypothesized value ta0.
The samples are independent, and we are not able to assume normality for the data, either due to lack of enough 
sample for Central limit theorem to kick in or having non interval data (such as ordinal data). We assume the observations on each
side of median follow binomial distribution. We use the same approach similar to population proportion with p=0.5.

Hypothesis:
    H0:	The median of the population is equal to ta0
    Ha:	The median of the population is not equal to ta0

Test Statistic:
    Z0 = (S-ta0)/sd
    sd = sqrt( p(1-p)n ) = 0.5sqrt(n)
    S: the count of deviation from the median

Distribution:
    Z - two-tailed test
* Note that we always use the positive side of the Z distribution since
S is the larger value of the deviation below or above the median

E.g.: It is claimed that the median of the population is equal to 42
'''
# observations
obs = np.array([41.83, 41.01, 42.68, 41.37, 41.83, 40.50, 41.70, 41.42, 43.01, 41.56, 41.89, 42.56])
# the count of deviation that are less than the median
S = np.maximum(np.sum(obs<42), np.sum(obs>42))
# number of observations
n = len(obs)
# test statistics
test_statistics_Z0 = (S-0.5*n)/(0.5*np.sqrt(n))
# p_value on the two tail test
from scipy.stats import norm
p_value = 2*(1-norm.cdf(test_statistics_Z0, loc=0, scale=1))

'''
Hypothesis:
    H0:	The median of the population is greater than equal to ta0
    Ha:	The median of the population is less than ta0

Test Statistic:
    S: the count of deviation that are below the median

Distribution:
    Z - one-tailed test (upper) since we are measuring the negative values
* Note that we always use the positive side of the Z distribution since
S is the larger value of the deviations from the median

E.g.: It is claimed that the median of the population is greater than equal to 2.5
'''
# observations
obs = np.array([2.4, 2.5, 1.7, 1.6, 1.9, 2.6, 1.3, 1.9, 2.0, 2.6, 2.3, 2.0, 1.8, 2.7, 1.7, 1.9])
# the count of deviation that are less than the median
S = np.sum(obs<2.5)
# number of observations
n = len(obs)
# test statistics
test_statistics_Z0 = (S-0.5*n)/(0.5*np.sqrt(n))
# p_value on the upper tail test
from scipy.stats import norm
p_value = 1-norm.cdf(test_statistics_Z0, loc=0, scale=1)

'''
Hypothesis:
    H0:	The median of the population is less than equal to ta0
    Ha:	The median of the population is greater than ta0

Test Statistic:
    S: the count of deviation that are above the median

Distribution:
    Z - one-tailed test (upper) since we are measuring the positive values
* Note that we always use the positive side of the Z distribution since
S is the larger value of the deviations from the median

E.g.: It is claimed that the median of the population is less than equal to 2.5
'''
# observations
obs = np.array([41.83, 43.01, 42.68, 41.37, 43.83, 42.50, 41.70, 41.42, 43.01, 42.56, 41.89, 42.56])
# the count of deviation that are less than the median
S = np.sum(obs>42)
# number of observations
n = len(obs)
# test statistics
test_statistics_Z0 = (S-0.5*n)/(0.5*np.sqrt(n))
# p_value on the upper tail test
from scipy.stats import norm
p_value = 1-norm.cdf(test_statistics_Z0, loc=0, scale=1)



#==============================================================================
#==============================================================================
# Wilcoxon Rank Sum Test (Nonparametric) - Test if the central tendency of two population are similar or not
#==============================================================================
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html
'''
Purpose: can be used to see if the central tendency of two populations (D1 and D2) are similar or one is shfited to the left or right of the other one.
The samples are independent, and we are not able to assume normality for the data, either due to lack of enough 
sample for Central limit theorem to kick in or having non interval data (such as ordinal data).
We should have atleast 10 samples from each population (n1 and n2 >10)

Hypothesis:
    H0:	D1 and D2 are identical
    Ha:	D1 is shifted either to the left or right of D2
    
Test Statistic:
    Z0 = ( T1-((n1*n2+n1*(n1+1))/2) )/np.sqrt((n1*n2*(n1+n2+1))/12)
    T1: sum of the ranks of samples from population 1
    T2: sum of the ranks of samples from population 2

Distribution:
    Z - two-tailed test

E.g.: It is claimed that the two populations have similar central tendencies
'''
obs1__ = [210, 212, 211, 211, 190, 213, 179, 217, 164, 217]
obs2__ = [216, 209, 162, 137, 211, 216, 212, 153, 152, 219]

# Using Scipy
from scipy.stats import ranksums
ranksums(obs1__, obs2__)

# Performing the test manually
n1 = len(obs1__)
n2 = len(obs2__)
# Keeping the track of samples
obs1 = [[obs1__[i],1] for i in range(n1)]
obs2 = [[obs2__[i],2] for i in range(n2)]
# Mixing the two samples together while keeping the track which observation belongs to which population
obs = obs1 + obs2
# Sorting the observations
obs.sort()
# Ordering the observations
obs = [[obs[i][0], obs[i][1], i+1] for i in range(n1+n2)]
# averaging the orders for ties
i = 0
while i < n1+n2-1:
    s = i
    count = 0
    while obs[i][0] == obs[i+1][0] and i+1<n1+n2:
        count += 1
        i += 1
        if i+1 >= n1+n2:
            break
        #end
    #end
    
    if count > 0:
        ave_ = (obs[s][2] + obs[s+count][2])/2
        for j in range(count+1):
            obs[s+j][2] = ave_
        #end
        i -= 1
    #end
    
    i += 1
#end
import numpy as np
obs = np.array(obs)
obs1_ = obs[obs[:,1]==1,2]
obs2_ = obs[obs[:,1]==2,2]
# sum of the ranks
T1 = np.sum(obs1_)
T2 = np.sum(obs2_)
# test statistics
test_statistics_Z0 = ( T1-((n1*n2+n1*(n1+1))/2) )/np.sqrt((n1*n2*(n1+n2+1))/12)
from scipy.stats import norm
p_value = np.minimum(2*norm.cdf(test_statistics_Z0, loc=0, scale=1), 2*(1-norm.cdf(test_statistics_Z0, loc=0, scale=1)) )

'''
Hypothesis:
    H0:	D1 and D2 are identical
    Ha:	D1 is shifted to the right of D2
    
Test Statistic:
    Z0 = ( T1-((n1*n2+n1*(n1+1))/2) )/np.sqrt((n1*n2*(n1+n2+1))/12)
    T1: sum of the ranks of samples from population 1
    T2: sum of the ranks of samples from population 2

Distribution:
    Z - One-tailed test (upper tail)

E.g.: It is claimed that the central tendency of population 1 is shifted to the right of population 2
'''
obs1__ = [210, 212, 211, 211, 190, 213, 179, 217, 164, 217]
obs2__ = [216, 209, 162, 137, 211, 216, 212, 153, 152, 219]

# Using Scipy
from scipy.stats import ranksums
ranksums(obs1__, obs2__, alternative='greater')

'''
Hypothesis:
    H0:	D1 and D2 are identical
    Ha:	D1 is shifted to the left of D2
    
Test Statistic:
    Z0 = ( T1-((n1*n2+n1*(n1+1))/2) )/np.sqrt((n1*n2*(n1+n2+1))/12)
    T1: sum of the ranks of samples from population 1
    T2: sum of the ranks of samples from population 2

Distribution:
    Z - One-tailed test (lower tail)

E.g.: It is claimed that the central tendency of population 1 is shifted to the left of population 2
'''
obs1__ = [210, 212, 211, 211, 190, 213, 179, 217, 164, 217]
obs2__ = [216, 209, 162, 137, 211, 216, 212, 153, 152, 219]

# Using Scipy
from scipy.stats import ranksums
ranksums(obs1__, obs2__, alternative='less')



#==============================================================================
#==============================================================================
# Mann-Whitney Test (Nonparametric) - Test for central tendency of a population that cannot be tested via parametric tests
#==============================================================================
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html
'''
Purpose: can be used to see if the central tendency of two populations (D1 and D2) are similar or one is shfited to the left or right of the other one.
The samples are independent, and we are not able to assume normality for the data, either due to lack of enough 
sample for Central limit theorem to kick in or having non interval data (such as ordinal data).

Hypothesis:
    H0:	D1 and D2 are identical
    Ha:	D1 is shifted either to the left or right of D2
    
Test Statistic:
    Z0 = ( T1-((n1*n2+n1*(n1+1))/2) )/np.sqrt((n1*n2*(n1+n2+1))/12)
    T1: sum of the ranks of samples from population 1
    T2: sum of the ranks of samples from population 2

Distribution:
    Z - two-tailed test

E.g.: It is claimed that the two populations have similar central tendencies
'''







#==============================================================================
#==============================================================================
# Pearson's Chi-Square Test - Test for goodness of fit
#==============================================================================
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
'''
Purpose: can be used to see if the data follow hypothesized distribution.

Hypothesis:
    H0:	data follow the hypothesized distribution
    Ha:	data do not follow the hypothesized distribution
    
Test Statistic:
    W = sum( (Oi-Ei)^2/Ei ) over all the bins or class
    Oi: the frequency of observations we collected in the i-th bin or class of a histogram
    Ei: the frequency of observations we'd expect in the i-th bin if the data follows the distribution we're testing for

Distribution:
    Chi2 - One-tailed test (upper tail) because the nominator is squared and always positive
    df: k-m-1
    k: number of bins or classes
    m: number of parameters in the distribution (e.g. 2 for normal)

E.g.: It is claimed that the following data follow an exponential distibution
'''
import numpy as np
# observed data
data = np.array([1476, 182, 246, 47, 210, 214, 467, 300, 499, 442, 438, 284, 428, 401, 98, 552, 20, 400, 
        553, 597, 210, 221, 1563, 796, 279, 767, 2025, 289, 157, 36, 31, 247, 1297, 185, 1024])
# finding the best fit parameter using the method of maximum likelihood
from scipy.stats import expon
initial, beta = expon.fit(data, floc=0, method="MLE")
lambda_ = 1/beta
# finding the bins and their ranges
n = data.shape[0]
n_bins = np.ceil(np.sqrt(n)).astype(int)
bins = np.zeros((n_bins,2))
bin_size = np.ceil((np.max(data)-np.min(data))/n_bins).astype(int)
bins[0] = np.array([np.floor(np.min(data)).astype(int), np.floor(np.min(data)).astype(int)+bin_size])
for i in range(1, n_bins):
    bins[i] = np.array([bins[i-1][1], bins[i-1][1]+bin_size])
#end
# counting the observed and expected value for each bins (lower bound included)
obs_exp = np.zeros((n_bins,2))
obs_exp[0] = np.array([ data[data<bins[0][1]].shape[0], n*(expon.cdf(bins[0][1], loc=initial, scale=beta)) ])
for i in range(1, n_bins-1):
    obs_exp[i] = np.array([ data[(data>=bins[i][0])*(data<bins[i][1])].shape[0], n*(expon.cdf(bins[i][1], loc=initial, scale=beta)-expon.cdf(bins[i][0], loc=initial, scale=beta)) ])
#end
obs_exp[n_bins-1] = np.array([ data[(data>=bins[n_bins-1][0])*(data<bins[n_bins-1][1])].shape[0], n*(1-expon.cdf(bins[n_bins-1][0], loc=initial, scale=beta)) ])
# calculating the test statistics
W = 0
for i in range(n_bins):
    W += (obs_exp[i][0]-obs_exp[i][1])**2/obs_exp[i][1]
#end
# degree of freedom
df = n_bins-1-1
from scipy.stats import chi2
p_value = 1-chi2.cdf(W, df, loc=0)

# performing test statistics using the Scipy function
from scipy.stats import chisquare
chisquare(obs_exp[:,0], f_exp=obs_exp[:,1], ddof=1) # ddof is the number of parameters (df=k - 1 - ddof)



#==============================================================================
#==============================================================================
# Anderson-Darling test - Test if data coming from a particular distribution
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
'''
Purpose: can be used to see if data is coming from a particular distribution

Hypothesis:
    H0:	The data follow a specified distribution            test_statistics < critical_value    ---->    We do not have enough evidence to reject H0
    Ha:	The data do not follow the specified distribution   test_statistics > critical_value    ---->    We have enough evidence to reject H0
    
    Distribution Options: ‘norm’, ‘expon’, ‘logistic’, ‘gumbel’, ‘gumbel_l’, ‘gumbel_r’, ‘extreme1’
'''
# importing package
from scipy.stats import anderson
# generating normal samples
from scipy.stats import norm
data = norm.rvs(loc=100, scale=1, size=1000)
#
results = anderson(data, dist='norm')
# the value of test statistics
statistic = results[0]
# critical values associated with various significance levels
critical_values = results[1]
# significance level in percentile (%) for several values
significance_level = results[2] # (%)







#==============================================================================
#==============================================================================
# Bartlett's test - Test if data coming from an Exponential distribution
#==============================================================================
# 
'''
Purpose: can be used to see if data is coming from an Exponential distribution

Hypothesis:
    H0:	The data follow an Exponential distribution
    Ha:	The data do not follow an Exponential distribution

Test Statistic:
    Br = 2*r*[ ln(sum(ti)/r) - (1/r)*sum(ln(ti)) ]/[ 1+(r+1)/(6*r) ] over all of observations
    t1: first observation (e.g., time of first failure)
    ti: distance between the (i-1)th and i-th observation
    r: total number of observations during the test

Distribution:
    Chi2 - One-tailed test (upper tail) because the nominator is squared and always positive
    df: r-1

E.g.: It is claimed that the following data follow an exponential distibution
'''
import numpy as np
# observed data
data = np.array([1476, 182, 246, 47, 210, 214, 467, 300, 499, 442, 438, 284, 428, 401, 98, 552, 20, 400, 
        553, 597, 210, 221, 1563, 796, 279, 767, 2025, 289, 157, 36, 31, 247, 1297, 185, 1024])

























# Shapiro-Wilk test for normality
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
'''
Purpose: can be used to see if data is normaly distributed

Hypothesis:
    H0:	The data was drawn from a normal distribution           p_value < desired_p_value    ---->    We do not have enough evidence to reject H0
    Ha:	The data was not drawn from a normal distribution       p_value > desired_p_value    ---->    We have enough evidence to reject H0
'''

# importing package
from scipy.stats import shapiro

# generating normal samples
from scipy.stats import norm
data = norm.rvs(loc=100, scale=1, size=1000)

results = shapiro(data)

# the value of test statistics
statistic = results[0]

# the p-value associated with the test statistics
p_value = results[1]



# Pearson chi square test to check if the categorical data has the given frequencies
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
'''
Purpose: can be used to see if  the categorical data has the given frequencies

Hypothesis:
    H0:	The categorical data has the given frequencies      p_value < desired_p_value    ---->    We do not have enough evidence to reject H0
    Ha:	Otherwise                                           p_value > desired_p_value    ---->    We have enough evidence to reject H0
'''

# importing package
from scipy.stats import chisquare

# generating normal samples
from scipy.stats import norm
data = norm.rvs(loc=100, scale=1, size=1000)

results = shapiro(data)

# the value of test statistics
statistic = results[0]

# the p-value associated with the test statistics
p_value = results[1]




# Kolmogorov-Smirnov test for goodness of fit test (normal)
#==============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html#scipy.stats.kstest
'''
Purpose: can be used to see if data is normaly distributed

Hypothesis (two-sided):
    H0:	The two distributions are identical, F(x)=G(x)          test_statistics < critical_value    ---->    We do not have enough evidence to reject H0
    Ha:	They are not identical                                  test_statistics > critical_value    ---->    We have enough evidence to reject H0
    
Hypothesis (less):
    H0:	F(x) >= G(x) for all x                                  test_statistics < critical_value    ---->    We do not have enough evidence to reject H0
    Ha:	F(x) < G(x) for at least one x                          test_statistics > critical_value    ---->    We have enough evidence to reject H0
    
Hypothesis (greater):
    H0:	F(x) <= G(x) for all x                                  test_statistics < critical_value    ---->    We do not have enough evidence to reject H0
    Ha:	F(x) > G(x) for at least one x                          test_statistics > critical_value    ---->    We have enough evidence to reject H0

    Distribution Options: ‘norm’, ‘expon’, ‘logistic’, ‘gumbel’, ‘gumbel_l’, ‘gumbel_r’, ‘extreme1’
'''

# importing package
from scipy.stats import kstest

# generating normal samples
from scipy.stats import norm
data1 = norm.rvs(loc=100, scale=1, size=1000)

# Not done










results = anderson(r)

# the value of test statistics
statistic = results[0]

# critical values associated with various significance levels
critical_values = results[1]

# significance level in percentile (%) for seevral values
significance_level = results[2] # (%)





















# parameters
mean = 1 # mean (loc in the scipy model)
sd = 1 # standard deviation (scale in the scipy model)

# Calculate the first four moments: Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’): use any combination of 'mvsk' to generate each
mean, var, skew, kurt = norm.stats(loc=mean, scale=sd, moments='mvsk')

# Probability density function - P(y)
fig, ax = plt.subplots(1, 1)
# The range of values that cover from 0.01 and 0.99
x = np.linspace(norm.ppf(0.01, loc=mean, scale=sd), norm.ppf(0.99, loc=mean, scale=sd), 100) # ppf(0.99, loc=mean, scale=sd) is Percent point function (inverse of cdf — percentiles)
# plotting the distribution
ax.plot(x, norm.pdf(x, loc=mean, scale=sd), 'r-', lw=3, alpha=0.6, label='norm pdf') # pdf(x, loc=mean, scale=sd) is Probability density function
plt.show()

# Cumulative distribution function F(Y<=k)
F_x = norm.cdf(0, loc=mean, scale=sd) # cdf(x, loc=mean, scale=sd) is Cumulative distribution function

# Generate random numbers
r = norm.rvs(loc=mean, scale=sd, size=1000)

# pdf on top of histogram
fig, ax = plt.subplots(1, 1)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.plot(x, norm.pdf(x, loc=mean, scale=sd), 'r-', lw=3, alpha=0.6, label='norm pdf')
ax.legend(loc='best', frameon=False)
plt.show()
'''
Author: Philippos Papaphilippou
'''

import math
import scipy.stats
import numpy as np
import math
from scipy.optimize import least_squares
from skew_normal import cdf_skewnormal

'''
Confidence Interval
The formula is from paper:
Vollset, S.E., 1993. Confidence intervals for a binomial proportion. Statistics in medicine, 12(9), pp.809-824.

More information:
Newcombe, Robert G. "Two-Sided Confidence Intervals for the Single Proportion: Comparison of Seven Methods," Statistics in Medicine, 17, 857-872 (1998).
http://vassarstats.net/prop1.html
https://www.medcalc.org/manual/values_of_the_normal_distribution.php
http://slideplayer.com/slide/5055000/
'''


def wald_interval(accepted, total, quantile_prob):
    x = float(accepted)
    n = float(total)
    c = scipy.stats.norm.ppf(quantile_prob)

    if (x > n):
        print("Someting went wrong inside wald_interval()")
        exit(1)

    mean = x/n
    interval = c/math.sqrt(n) * math.sqrt((x/n)*(1-x/n)) + 1/(2*n)

    lower = max(0, mean - interval)
    higher = min(1, mean + interval)
    conf_int = higher - lower

    return conf_int, lower, higher


def conf_int_N_draws(mean, stdev, N, quantile_prob):
    stdev_of_mean = stdev/math.sqrt(N)
    result = scipy.stats.norm.interval(
        quantile_prob, loc=mean, scale=stdev_of_mean)
    lower = max(0, result[0])
    higher = min(1, result[1])
    conf_int = higher - lower

    return conf_int, lower, higher


'''	
- For the CDF of the skew normal distribution I used skew_normal.py from
  http://azzalini.stat.unipd.it/SN/

  The way I use Skew_normal contains a bug workaround. For shapes in range [-1,1], 
  such as skew_normal.cdf_skewnormal(10,shape=0) or ... (-10000,shape=-1), it returns nan.
  When it returns a NaN, I return 0 because this seems to be the case. 
  Ideally the model_to_fit function would be the one-liner: skewnorm.cdf((x+a[0])*a[1], a[2]) (see below)
'''


def model_to_fit(a, x):
    values = []
    x_ = np.asarray(x)
    if x_.ndim > 0:
        for x0 in x_:
            value = max(0, cdf_skewnormal((x0+a[0])*a[1], shape=a[2])[0])
            if not math.isnan(value):
                values.append(value)
            else:
                values.append(0.0)
        return np.array(values)
    else:
        value = max(0, cdf_skewnormal((x+a[0])*a[1], shape=a[2])[0])
        if not math.isnan(value):
            return value
        else:
            return 0.0


def function(a, x, y):
    return model_to_fit(a, x) - y


'''
- The cdf function of scipy.stats.skewnorm has a bug. For large x values it returns
  0 instead of 1. E.g. skewnorm.cdf(100,0). This is the reason I used skew_normal.py.
- Link to the bug https://github.com/scipy/scipy/issues/7746	
'''
#from scipy.stats import skewnorm
# def model_to_fit(a, x):
#    return skewnorm.cdf((x+a[0])*a[1], a[2])

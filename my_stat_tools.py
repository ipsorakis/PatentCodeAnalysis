import numpy
import scipy
import scipy.signal
import math

def correlation(x,y):
    T = len(x)
    res = 0
    for t in range(0,T):
        if not numpy.isnan(x[t]) and not numpy.isnan(y[t]):
            res += x[t] * y[t]
    return  res

def cross_correlation(x,y,time_lag):
    T = len(x)
    res = 0
    non_nan_ts = 0
    for t in range(0,T):
        if 0<= t+time_lag <T and not numpy.isnan(x[t]) and not numpy.isnan(y[t+time_lag]):
            non_nan_ts+=1
            res += x[t]*y[t+time_lag]
    return  res

def normalised_correlation(x,y):
    T = len(x)

    # calculate mean and std
    x_nn_list=[]
    y_nn_list=[]
    for t in range(0,T):
        if not numpy.isnan(x[t]) and not numpy.isnan(y[t]):
            x_nn_list.append(x[t])
            y_nn_list.append(y[t])

    meanx = numpy.mean(x_nn_list)
    meany = numpy.mean(y_nn_list)
    stdx = numpy.std(x_nn_list)
    stdy = numpy.std(y_nn_list)

    # calculate normalised correlation
    if stdx==0 or stdy ==0:
        return numpy.nan
    else:
        res = 0
        non_nan_ts = 0
        for t in range(0,T):
            if not numpy.isnan(x[t]) and not numpy.isnan(y[t]):
                non_nan_ts+=1
                res += (x[t] - meanx) * (y[t] - meany)

        res  /= (non_nan_ts * stdx * stdy)

        return  res

def normalised_cross_correlation(x,y,time_lag):
    T = len(x)

    # calculate mean and std
    x_nn_list=[]
    y_nn_list=[]
    for t in range(0,T):
        if 0<= t+time_lag <T and not numpy.isnan(x[t]) and not numpy.isnan(y[t+time_lag]):
            x_nn_list.append(x[t])
            y_nn_list.append(y[t+time_lag])

    meanx = numpy.mean(x_nn_list)
    meany = numpy.mean(y_nn_list)
    stdx = numpy.std(x_nn_list)
    stdy = numpy.std(y_nn_list)

    if stdx==0 or stdy ==0:
        return numpy.nan
    else:
        res = 0
        non_nan_ts = 0
        for t in range(0,T):
            if 0<= t+time_lag <T and not numpy.isnan(x[t]) and not numpy.isnan(y[t+time_lag]):
                non_nan_ts+=1
                res += (x[t] - meanx) * (y[t+time_lag] - meany)

        res  /= (non_nan_ts * stdx * stdy)

        return  res

def cross_correlation_similarity(c,max_lag,sigma=1):
    aux = 0
    wsum = 0
    t = -1
    for tau in range(-max_lag+1,max_lag):
        t+=1
        wt = scipy.stats.norm.pdf(tau,0,sigma)
        aux +=  wt*c[t]
        wsum +=wt

    return aux / wsum

def time_series_flatness(X): # normalised entropy of activity
    T = len(X)

    sumx = 1.*X.nansum()

    H = 0
    num_nonans = 0
    for t in range(0,T):
        if X[t] is not numpy.nan:
            num_nonans+=1
            pi = X[t]/sumx
            H += -(pi!=0)*(pi*numpy.log(pi))

    return H / numpy.log(num_nonans)

def isnan(x):
    return x is numpy.nan or math.isnan(x)

def nan_to_zero(x):
    N = len(x)
    xnnan = numpy.zeros(N)

    for n in range(0,N):
        if not isnan(x[n]):
            xnnan[n] = x[n]

    return xnnan

def get_entropy(p):
    H = 0
    for pi in p:
        H += -(pi!=0)*(pi*numpy.log(pi))
    return H

def get_normalised_entropy(p):
    H = get_entropy(p)
    N = len(p)
    return H / numpy.log(N)
# Import some python packages we'll use
import pylab as plt
import numpy as np
import scipy.stats
import scipy.optimize

np.float = np.float64

# Example data set from  arxiv:1008.4686, table 1 (https://arxiv.org/abs/1008.4686)
# You can also refer to that paper for more background, equations, etc.
alldata = np.array([[201,592,61],[244,401,25],[47 ,583,38],[287,402,15],[203,495,21],[58 ,173,15],[210,479,27],
                   [202,504,14],[198,510,30],[158,416,16],[165,393,14],[201,442,25],[157,317,52],[131,311,16],
                   [166,400,34],[160,337,31],[186,423,42],[125,334,26],[218,533,16],[146,344,22],]).astype(float)
# The first 5 data points are outliers; for the first part we'll just use the "good" data points
x    = alldata[5:,0]
y    = alldata[5:,1]
# this is the standard deviation (uncertainty) on the y measurements, also known as \sigma_i
yerr = alldata[5:,2]
# Note that x, y, and yerr are *vectors*

def log_likelihood_one(params, x, y, yerr):
    '''This function computes the log-likelihood of a data set with coordinates
    (x_i,y_i) and Gaussian uncertainties on y_i of yerr_i (aka sigma_i)

    The model is a straight line, so the model's predicted y values are
        y_pred_i = b + m x_i.

    params = (b,m) are the parameters (scalars)
    x,y,yerr are arrays (aka vectors)

    Return value is a scalar log-likelihood.
    '''
    # unpack the parameters
    b,m = params
    # compute the vector y_pred, the model predictions for the y measurements
    y_pred = b + m * x
    # compute the log-likelihoods for the individual data points
    # (the quantity inside the sum in the text above)
    loglikes = np.log(1. / (np.sqrt(2. * np.pi) * yerr)) + -0.5*(y - y_pred)**2 / yerr**2
    # the log-likelihood for the whole vector of measurements is the sum of individual log-likelihoods
    loglike = np.sum(loglikes)
    return loglike

def neg_ll_one(params, x, y, yerr):
    return -log_likelihood_one(params, x, y, yerr)

# The optimizer we're using here requires an initial guess.  This log-likelihood happens
# to be pretty simple, so we don't need to work very hard to give it a good initial guess!
initial_params = [0., 0.]
# The "args" parameter here gets passed to the neg_ll_one function (after the parameters)
R = scipy.optimize.minimize(neg_ll_one, initial_params, args=(x, y, yerr))
R

# The optimizer gives us the parameters that maximize the log-likelihood, along with an estimate of the uncertainties.
# These are the maximum-likelihood values from the optimizer
b_ml,m_ml = R.x
xx = np.linspace(50, 250, 50)
# Draw a sampling of B,M parameter values that are consistent with the fit,
# using the estimated inverse-Hessian matrix (parameter covariance)
BM = scipy.stats.multivariate_normal.rvs(mean=R.x, cov=R.hess_inv, size=20)

# You can also plot the ellipse showing the constraints in B,M space by manipulating hess_inv:
U,s,V = np.linalg.svd(R.hess_inv)
S = np.dot(U, np.diag(np.sqrt(s)))
th = np.linspace(0,2.*np.pi,200)
xy = np.vstack((np.sin(th), np.cos(th)))
dbm = np.dot(S, xy).T
ellipse_b = R.x[0] + dbm[:,0]
ellipse_m = R.x[1] + dbm[:,1]

def log_posterior_one(params, args):
    (x, y, yerr) = args
    loglike = log_likelihood_one(params, x, y, yerr)
    # Improper, flat priors on params!
    logprior = 0.
    return loglike + logprior

import emcee

ndim, nwalkers = 2, 30
p0 = np.random.uniform(size=(nwalkers, ndim))
p0[:,0] *= 4.
p0[:,0] += 0.
p0[:,1] *= 0.02
p0[:,1] += 2.0

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_one, args=([x,y,yerr],))
sampler.run_mcmc(p0, 100);
sampler.chain.shape

def log_post_save(x, all_params, *args):
    all_params.append(x)
    return log_posterior_one(x, *args)

np.random.seed(40004)
ndim, nwalkers = 2, 30
p0 = np.random.uniform(size=(nwalkers, ndim))
p0[:,0] *= 4.
p0[:,0] += 0.
p0[:,1] *= 0.02
p0[:,1] += 2.0

all_params = []
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_save, args=(all_params, [x,y,yerr],))

state = p0

plt.figure(figsize=(6,6))
for step in range(40):
    all_params.clear()
    
    state = list(sampler.sample(state, store=True))[0]
    
    plt.clf()
    for i in range(1,5):
        eb = R.x[0] + i*dbm[:,0]
        em = R.x[1] + i*dbm[:,1]
        plt.plot(eb, em, 'k-', alpha=0.3)
    plt.axis([-25, 100, 1.95, 2.5]);
    ch = sampler.chain
    #print('Chain:', ch.shape)
    ap = np.vstack(all_params)
    plt.plot(ch[:,step,0], sampler.chain[:,step,1], '.');
    plt.title('Step %i' % step)
    plt.savefig('emcee/emcee-%i.png' % step)
    plt.plot(ap[:,0], ap[:,1], 'r.');
    plt.plot(ch[:,step,0], sampler.chain[:,step,1], '.', color='#1f77b4');
    plt.savefig('emcee/emcee-B-%i.png' % step)    

for i in range(40):
    print('\only<%i>{\includegraphics[height=0.8\\textheight]{emcee/emcee-%i.png}}%%' % (i+1, i))


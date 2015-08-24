"""
Some commonly used functions for defining a task.

"""
from __future__ import division

import numpy as np

#-----------------------------------------------------------------------------------------
# Define E/I populations
#-----------------------------------------------------------------------------------------

def generate_ei(N, pE=0.8):
    """
    E/I signature.

    Parameters
    ----------

    N : int
        Number of recurrent units.

    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.

    """
    assert 0 <= pE <= 1

    Nexc = int(pE*N)
    Ninh = N - Nexc

    idx = range(N)
    EXC = idx[:Nexc]
    INH = idx[Nexc:]

    ei       = np.ones(N, dtype=int)
    ei[INH] *= -1

    return ei, EXC, INH

#-----------------------------------------------------------------------------------------
# Functions for defining task epochs
#-----------------------------------------------------------------------------------------

def get_idx(t, interval):
    start, end = interval
    
    return list(np.where((start < t) & (t <= end))[0])

def get_epochs_idx(dt, epochs):
    t = np.linspace(dt, epochs['Tf'], int(epochs['Tf']/dt))
    assert t[1] - t[0] == dt, "[ tasktools.get_epochs_idx ] dt doesn't fit into Tf."

    return t, {k: get_idx(t, v) for k, v in epochs.items() if k != 'Tf'}

#-----------------------------------------------------------------------------------------
# Functions for generating epoch durations that are multiples of the time step
#-----------------------------------------------------------------------------------------

def uniform(rng, dt, xmin, xmax):
    return (rng.uniform(xmin, xmax)//dt)*dt

def truncated_exponential(rng, dt, mean, xmin=0, xmax=np.inf):
    while True:
        x = rng.exponential(mean)
        if xmin <= x < xmax:
            return (x//dt)*dt

def truncated_normal(rng, dt, mean, sigma, xmin=-np.inf, xmax=np.inf):
    while True:
        x = rng.normal(mean, sigma)
        if xmin <= x < xmax:
            return (x//dt)*dt

#-----------------------------------------------------------------------------------------
# Functions for generating orientation tuning curves
#-----------------------------------------------------------------------------------------

def deg2rad(s):
    return s*np.pi/180

def vonMises(s, spref, g=1, kappa=5, b=0, convert=True):
    arg = s - spref
    if convert:
        arg = deg2rad(arg)

    return g*np.exp(kappa*(np.cos(arg)-1)) + b

#-----------------------------------------------------------------------------------------
# Convert batch index to condition
#-----------------------------------------------------------------------------------------

def unravel_index(b, dims):
    return np.unravel_index(b, dims, order='F')

#-----------------------------------------------------------------------------------------
# Functions for generating connection matrices
#-----------------------------------------------------------------------------------------

def generate_Crec(ei, p_exc=1, p_inh=1, rng=None, seed=1, allow_self=False):
    if rng is None:
        rng = np.random.RandomState(seed)

    N    = len(ei)
    exc, = np.where(ei > 0)
    inh, = np.where(ei < 0)

    C = np.zeros((N, N))
    for i in xrange(N):
        C[i,exc]  = 1*(rng.uniform(size=len(exc)) < p_exc)
        C[i,inh]  = 1*(rng.uniform(size=len(inh)) < p_inh)
        C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])

    if not allow_self:
        np.fill_diagonal(C, 0)

    return C

#-----------------------------------------------------------------------------------------
# Performance measure
#-----------------------------------------------------------------------------------------

def performance_2afc(trials, z):
    ends    = [len(trial['t'])-1 for trial in trials]
    choices = [np.argmax(z[ends[i],i]) for i, end in enumerate(ends)]
    correct = [choice == trial['info']['choice']
               for choice, trial in zip(choices, trials) if trial['info']]

    return 100*sum(correct)/len(correct)

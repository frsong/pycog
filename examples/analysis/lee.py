"""
Analyze variants of the Lee sequence generation task.

"""
from __future__ import division

import cPickle as pickle
import os
import sys

import numpy as np

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure, PCA

THIS = "examples.analysis.lee"

#=========================================================================================
# Setup
#=========================================================================================

# File to store trials in
def get_trialsfile(p):
    return '{}/{}_trials.pkl'.format(p['trialspath'], p['name'])

# Load trials
def load_trials(trialsfile):
    with open(trialsfile) as f:
        trials = pickle.load(f)

    return trials, len(trials)

#=========================================================================================

def run_trials(p, args):
    """
    Run trials.

    """
    # Model
    m = p['model']

    # Number of trials
    try:
        ntrials = int(args[0])
    except:
        ntrials = 1
    ntrials *= m.nseq

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            b = i % m.nseq
            if b == 0:
                if not trials:
                    seqs = range(m.nseq)
                else:
                    seqs = rng.permutation(m.nseq)

            # Sequence number
            seq = seqs[b] + 1

            # Trial
            trial_func = m.generate_trial
            trial_args = {'name': 'test', 'seq': seq}
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            s = "Trial {:>{}}/{}: Sequence #{}".format(i+1, w, ntrials, info['seq'])
            sys.stdout.write(backspaces*'\b' + s)
            sys.stdout.flush()
            backspaces = len(s)

            # Add
            dt    = rnn.t[1] - rnn.t[0]
            step  = int(p['dt_save']/dt)
            trial = {
                't':    rnn.t[::step],
                'u':    rnn.u[:,::step],
                'r':    rnn.r[:,::step],
                'z':    rnn.z[:,::step],
                'info': info
                }
            trials.append(trial)
    except KeyboardInterrupt:
        pass
    print("")

    # Save all
    filename = get_trialsfile(p)
    with open(filename, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(filename)*1e-9
    print("[ {}.run_trials ] Trials saved to {} ({:.1f} GB)".format(THIS, filename, size))

#=========================================================================================

def pca_analysis(trials, min_std=0.1):
    """
    Perform PCA analysis.

    Parameters
    ----------

    trials : dict
             Dictionary of `seq: trial` pairs.

    min_std : float, optional
              Threshold for active unit.

    Returns
    -------

    pca : PCA object

    """
    #-------------------------------------------------------------------------------------
    # Build data matrix for PCA
    #-------------------------------------------------------------------------------------

    rows = trials.itervalues().next()['r'].shape[0]
    cols = sum([trial['r'].shape[1] for trial in trials.values()])

    X = np.zeros((rows, cols))
    k = 0
    for i, seq in enumerate(trials.keys()):
        r = trials[seq]['r']
        X[:,k:k+r.shape[1]] = r
        k += r.shape[1]
    X = X.T

    # Use only "active" units for PCA
    active_units = []
    for i in xrange(X.shape[1]):
        if np.std(X[:,i]) > min_std:
            active_units.append(i)
    X = X[:,active_units]
    print("[ lee.pca_analysis ] Number of active units: {}".format(len(active_units)))

    #-------------------------------------------------------------------------------------

    return PCA(X), active_units

#=========================================================================================

def do(action, args, p):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #-------------------------------------------------------------------------------------
    # Trials
    #-------------------------------------------------------------------------------------

    if action == 'trials':
        run_trials(p, args)

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action.".format(THIS))

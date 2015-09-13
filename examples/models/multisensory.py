"""
A multisensory integration task, loosely inspired by

  A category-free neural population supports evolving demands during decision-making.
  D. Raposo, M. T. Kaufman, & A. K. Churchland, Nature Neurosci. 2014.

  http://www.nature.com/neuro/journal/v17/n12/full/nn.3865.html

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 4
N    = 150
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)
Ne = len(EXC)
Ni = len(INH)

# Input labels
VISUAL_P   = 0 # Positively tuned visual input
AUDITORY_P = 1 # Positively tuned auditory input
VISUAL_N   = 2 # Negatively tuned visual input
AUDITORY_N = 3 # Negatively tuned auditory input

# Units receiving visual input
EXC_VISUAL = EXC[:Ne//3]
INH_VISUAL = INH[:Ni//3]

# Units receiving auditory input
EXC_AUDITORY = EXC[Ne//3:Ne*2//3]
INH_AUDITORY = INH[Ni//3:Ni*2//3]

#-----------------------------------------------------------------------------------------
# Input connectivity
#-----------------------------------------------------------------------------------------

Cin = np.zeros((N, Nin))
Cin[EXC_VISUAL   + INH_VISUAL,   VISUAL_P]   = 1
Cin[EXC_VISUAL   + INH_VISUAL,   VISUAL_N]   = 1
Cin[EXC_AUDITORY + INH_AUDITORY, AUDITORY_P] = 1
Cin[EXC_AUDITORY + INH_AUDITORY, AUDITORY_N] = 1

#-----------------------------------------------------------------------------------------
# Output connectivity: read out from excitatory units only
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

baseline_in = 0.2

var_in  = 0.01**2
var_rec = 0.15**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

modalities  = ['v', 'a', 'va']
freqs       = range(9, 16+1)
boundary    = 12.5
nconditions = len(modalities)*len(freqs)
pcatch      = 1/(nconditions + 1)

training_freqs = [6, 7, 8] + freqs + [17, 18, 19]

fmin = min(training_freqs)
fmax = max(training_freqs)

def scale_v_p(f):
    return 0.6 + 0.4*(f - fmin)/(fmax - fmin)

def scale_a_p(f):
    return 0.6 + 0.4*(f - fmin)/(fmax - fmin)

def scale_v_n(f):
    return 0.6 + 0.4*(fmax - f)/(fmax - fmin)

def scale_a_n(f):
    return 0.6 + 0.4*(fmax - f)/(fmax - fmin)

def generate_trial(rng, dt, params):
    #-------------------------------------------------------------------------------------
    # Select task condition
    #-------------------------------------------------------------------------------------

    catch_trial = False
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch):
            catch_trial = True
        else:
            modality = params.get('modality', rng.choice(modalities))
            freq     = params.get('freq',     rng.choice(training_freqs))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (len(modalities)*len(training_freqs) + 1)
        if b == 0:
            catch_trial = True
        else:
            k1, k2   = tasktools.unravel_index(b-1, (len(modalities), len(training_freqs)))
            modality = modalities[k1]
            freq     = training_freqs[k2]
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 1000}
    else:
        if params['name'] == 'test':
            fixation = 500
        else:
            fixation = 100
        stimulus = 1000
        decision = 300
        T       = fixation + stimulus + decision

        epochs = {
            'fixation': (0, fixation),
            'stimulus': (fixation, fixation + stimulus),
            'decision': (fixation + stimulus, T)
            }
        epochs['T'] = T

    #-------------------------------------------------------------------------------------
    # Trial info
    #-------------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        if freq > boundary:
            choice = 0
        else:
            choice = 1

        # Trial info
        trial['info'] = {'modality':  modality, 'freq': freq, 'choice': choice}

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        if 'v' in modality:
            X[e['stimulus'],VISUAL_P] = scale_v_p(freq)
            X[e['stimulus'],VISUAL_N] = scale_v_n(freq)
        if 'a' in modality:
            X[e['stimulus'],AUDITORY_P] = scale_a_p(freq)
            X[e['stimulus'],AUDITORY_N] = scale_a_n(freq)
    trial['inputs'] = X

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.zeros_like(Y)         # Mask matrix

        # Hold values
        hi = 1
        lo = 0.2

        if catch_trial:
            Y[:] = lo
            M[:] = 1
        else:
            # Fixation
            Y[e['fixation'],:] = lo

            # Decision
            Y[e['decision'],choice]   = hi
            Y[e['decision'],1-choice] = lo

            # Only care about fixation and decision periods
            M[e['fixation']+e['decision'],:] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #-------------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc

# Termination criterion
def terminate(performance_history):
    return np.mean(performance_history[-5:]) > 85

# Validation dataset
n_validation = 100*(len(modalities)*len(training_freqs) + 1)

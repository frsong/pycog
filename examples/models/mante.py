"""
Context-dependent integration task, loosely inspired by

  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V. Mante, D. Sussillo, K. V. Shinoy, & W. T. Newsome, Nature 2013.

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 6
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_in  = 0.01**2
var_rec = 0.1**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

contexts    = ['m', 'c']
cohs        = [1, 3, 10]
left_rights = [1, -1]
nconditions = len(contexts)*(len(cohs)*len(left_rights))**2
pcatch      = 1/(nconditions + 1)

training_cohs        = cohs
nconditions_training = len(contexts)*(len(training_cohs)*len(left_rights))**2

SCALE = 5
def scale(coh):
    return (1 + SCALE*coh/100)/2

def generate_trial(rng, dt, params):
    #-------------------------------------------------------------------------------------
    # Select task condition
    #-------------------------------------------------------------------------------------

    catch_trial = False
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch):
            catch_trial = True
        else:
            # Context
            context = params.get('context', rng.choice(contexts))

            # Coherences
            coh_m = params.get('coh_m', rng.choice(training_cohs))
            coh_c = params.get('coh_c', rng.choice(training_cohs))

            # Left/right
            left_right_m = params.get('left_right_m', rng.choice(left_rights))
            left_right_c = params.get('left_right_c', rng.choice(left_rights))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions_training + 1)
        if b == 0:
            catch_trial = True
        else:
            k = tasktools.unravel_index(b-1, (len(contexts),
                                              len(training_cohs), len(training_cohs),
                                              len(left_rights), len(left_rights)))
            context      = contexts[k[0]]
            coh_m        = training_cohs[k[1]]
            coh_c        = training_cohs[k[2]]
            left_right_m = left_rights[k[3]]
            left_right_c = left_rights[k[4]]
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 1000}
    else:
        if params['name'] == 'test':
            fixation = 300
        else:
            fixation = 100
        stimulus = 800
        decision = 300
        T        = fixation + stimulus + decision

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
        if context == 'm':
            left_right = left_right_m
        else:
            left_right = left_right_c

        # Correct choice
        if left_right > 0:
            choice = 0
        else:
            choice = 1

        # Trial info
        trial['info'] = {
            'coh_m':        coh_m,
            'left_right_m': left_right_m, 
            'coh_c':        coh_c,
            'left_right_c': left_right_c, 
            'context':      context,
            'choice':       choice
            }

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        # Context
        if context == 'm':
            X[e['stimulus'],0] = 1
        else:
            X[e['stimulus'],1] = 1

        # Motion stimulus
        if left_right_m > 0:
            choice_m = 0
        else:
            choice_m = 1
        X[e['stimulus'],2+choice_m]     = scale(+coh_m)
        X[e['stimulus'],2+(1-choice_m)] = scale(-coh_m)

        # Colour stimulus
        if left_right_c > 0:
            choice_c = 0
        else:
            choice_c = 1
        X[e['stimulus'],4+choice_c]     = scale(+coh_c)
        X[e['stimulus'],4+(1-choice_c)] = scale(-coh_c)
    trial['inputs'] = X

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):
        # Output matrix
        Y = np.zeros((len(t), Nout))

        # Mask matrix
        M = np.zeros_like(Y)

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
def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-5:]) > 95

# Validation dataset
n_validation = 100*(nconditions_training + 1)

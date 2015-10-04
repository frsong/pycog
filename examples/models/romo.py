"""
A parametric working memory task, loosely based on the vibrotactile delayed
discrimination task.

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://www.jneurosci.org/content/30/28/9424.short

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 2
N    = 500
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Input labels
POS = 0
NEG = 1

#-----------------------------------------------------------------------------------------
# Recurrent connectivity
#-----------------------------------------------------------------------------------------

Crec = tasktools.generate_Crec(ei, p_exc=0.1, p_inh=0.5)

#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

fpairs      = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
gt_lts      = ['>', '<']
nconditions = len(fpairs)*len(gt_lts)
pcatch      = 1/(nconditions + 1)

fall = np.ravel(fpairs)
fmin = np.min(fall)
fmax = np.max(fall)

def scale_p(f):
    return 0.2 + 0.8*(f - fmin)/(fmax - fmin)

def scale_n(f):
    return 0.2 + 0.8*(fmax - f)/(fmax - fmin)

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------------

    catch_trial = False
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch):
            catch_trial = True
        else:
            fpair = params.get('fpair', fpairs[rng.choice(len(fpairs))])
            gt_lt = params.get('gt_lt', rng.choice(gt_lts))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (len(fpairs), len(gt_lts)))
            fpair  = fpairs[k0]
            gt_lt  = gt_lts[k1]
    else:
        raise ValueError("Unknown trial type.")

    #---------------------------------------------------------------------------------
    # Epochs
    #---------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 2000}
    else:
        if params['name'] == 'test':
            fixation = 500
        else:
            fixation = 100
        f1 = 500
        if params['name'] == 'test':
            delay = 3000
        else:
            delay = tasktools.uniform(rng, dt, 2500, 3500)
        f2       = 500
        decision = 300
        T        = fixation + f1 + delay + f2 + decision

        epochs = {
            'fixation': (0, fixation),
            'f1':       (fixation, fixation + f1),
            'delay':    (fixation + f1, fixation + f1 + delay),
            'f2':       (fixation + f1 + delay, fixation + f1 + delay + f2),
            'decision': (fixation + f1 + delay + f2, T)
            }
        epochs['T'] = T

    #---------------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        if gt_lt == '>':
            f1, f2 = fpair
            choice = 0
        else:
            f2, f1 = fpair
            choice = 1

        # Info
        trial['info'] = {'f1': f1, 'f2': f2, 'choice': choice}

    #---------------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        # Stimulus 1
        X[e['f1'],POS] = scale_p(f1)
        X[e['f1'],NEG] = scale_n(f1)

        # Stimulus 2
        X[e['f2'],POS] = scale_p(f2)
        X[e['f2'],NEG] = scale_n(f2)
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

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

            # Mask
            M[e['fixation']+e['decision'],:] = 1

        trial['outputs'] = Y
        trial['mask']    = M

    #---------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc_min_condition

# Termination criterion
TARGET_PERFORMANCE = 85
def terminate(performance_history):
    return np.mean(performance_history[-5:]) > TARGET_PERFORMANCE

# Validation dataset
n_validation = 100*(nconditions + 1)

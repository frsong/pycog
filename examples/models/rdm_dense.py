"""
Perceptual decision-making task, loosely based on the random dot motion
discrimination task.

  Response of neurons in the lateral intraparietal area during a combined visual
  discrimination reaction time task.
  J. D. Roitman & M. N. Shadlen, JNS 2002.

  http://www.jneurosci.org/content/22/21/9475.abstract

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 2
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
# Task structure
#-----------------------------------------------------------------------------------------

cohs        = [1, 2, 4, 8, 16]
in_outs     = [1, -1]
nconditions = len(cohs)*len(in_outs)
pcatch      = 1/(nconditions + 1)

SCALE = 3.2
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
            coh    = params.get('coh',    rng.choice(cohs))
            in_out = params.get('in_out', rng.choice(in_outs))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (len(cohs), len(in_outs)))
            coh    = cohs[k0]
            in_out = in_outs[k1]
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 2000}
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
        # Correct choice
        if in_out > 0:
            choice = 0
        else:
            choice = 1

        # Trial info
        trial['info'] = {'coh': coh, 'in_out': in_out, 'choice': choice}

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        X[e['stimulus'],choice]   = scale(+coh)
        X[e['stimulus'],1-choice] = scale(-coh)
    trial['inputs'] = X

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output
        M = np.zeros_like(Y)         # Mask

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
TARGET_PERFORMANCE = 85
def terminate(performance_history):
    return np.mean(performance_history[-5:]) > TARGET_PERFORMANCE

# Validation
n_validation = 100*(nconditions + 1)

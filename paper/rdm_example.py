"""
Perceptual decision-making task, loosely based on the random dot motion task.

"""
from __future__ import division

import numpy as np

from pycog import Model, RNN, tasktools

#-------------------------------------------------------------------------------
# Network structure
#-------------------------------------------------------------------------------

Nin  = 2
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Output connectivity: read out from excitatory units only
Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-------------------------------------------------------------------------------
# Task structure
#-------------------------------------------------------------------------------

cohs        = [1, 2, 4, 8, 16]
in_outs     = [1, -1]
nconditions = len(cohs)*len(in_outs)
pcatch      = 1/(nconditions + 1)

SCALE = 3.2
def scale(coh):
    return (1 + SCALE*coh/100)/2

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------
    # Epochs
    #---------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 1000}
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

    #---------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # In discrete time
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

    #---------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        X[e['stimulus'],choice]   = scale(+coh)
        X[e['stimulus'],1-choice] = scale(-coh)
    trial['inputs'] = X

    #---------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output
        M = np.zeros_like(Y)         # Mask

        if catch_trial:
            Y[:] = 0.2
            M[:] = 1
        else:
            # Fixation
            Y[e['fixation'],:] = 0.2

            # Decision
            Y[e['decision'],choice]   = 1
            Y[e['decision'],1-choice] = 0.2

            # Only care about fixation and decision periods
            M[e['fixation']+e['decision'],:] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #---------------------------------------------------------------------------

    return trial

# Performance measure: two-alternative forced choice
performance = tasktools.performance_2afc

# Terminate training when psychometric performance exceeds 85%
def terminate(performance_history):
    return np.mean(performance_history[-5:]) > 85

# Validation dataset size
n_validation = 100*(nconditions + 1)

#-------------------------------------------------------------------------------
# Train model
#-------------------------------------------------------------------------------

model = Model(Nin=Nin, N=N, Nout=Nout, ei=ei, generate_trial=generate_trial,
              performance=performance, terminate=terminate,
              n_validation=n_validation)
model.train('savefile.pkl')

#-------------------------------------------------------------------------------
# Run the trained network with 51.2% coherence for choice 1
#-------------------------------------------------------------------------------

rnn        = RNN('savefile.pkl', {'dt': 0.5})
trial_func = generate_trial
trial_args = {'name': 'test', 'catch': False, 'coh': 16, 'in_out': 1}
info       = rnn.run(inputs=(trial_func, trial_args))

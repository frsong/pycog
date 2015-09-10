"""
The sequence generation part of

  Prefrontal neural correlates of memory for sequences.
  B. B. Averbeck & D. Lee, JNS 2007.

  http://www.jneurosci.org/content/27/9/2204.abstract

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Sequence-related definitions
#-----------------------------------------------------------------------------------------

# Sequence definitions
sequences = {
    1: [4] + [5, 1, 0],
    2: [4] + [3, 1, 2],
    3: [4] + [3, 7, 8],
    4: [4] + [5, 7, 6],
    5: [4] + [5, 1, 2],
    6: [4] + [3, 1, 0],
    7: [4] + [3, 7, 6],
    8: [4] + [5, 7, 8]
}
nseq = len(sequences)

# Possible targets from each position
#
#   0 1 2
#   3 4 5
#   6 7 8
#
options = {
    1: [0, 2],
    3: [1, 7],
    4: [3, 5],
    5: [1, 7],
    7: [6, 8]
}

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 9 + nseq
N    = 100
Nout = 2

# For addressing inputs
DOTS     = range(9)
SEQUENCE = range(9, 9+nseq)

# E/I
ei, EXC, INH = tasktools.generate_ei(N)
Nexc = len(EXC)
Ninh = len(INH)

# Inputs
EXC_SENSORY = EXC[:Nexc//2]
INH_SENSORY = INH[:Ninh//2]
EXC_MOTOR   = EXC[Nexc//2:]
INH_MOTOR   = INH[Ninh//2:]

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Input connectivity
#-----------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------
# Recurrent connectivity
#-----------------------------------------------------------------------------------------

Crec = tasktools.generate_Crec(ei, p_exc=1, p_inh=1)

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_in  = 0.01**2
var_rec = 0.01**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

# Screen coordinates
x0, y0 = -0.5, -0.5
dx, dy = +0.5, +0.5

# Convert dots to screen coordinates
def target_position(k):
    j = 2 - k//3
    i = k % 3

    return x0+i*dx, y0+j*dy

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------------

    if params['name'] in ['gradient', 'test']:
        seq = params.get('seq', rng.choice(sequences.keys()))
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % nseq
        if b == 0:
            generate_trial.seqs = rng.permutation(nseq)

        seq = generate_trial.seqs[b] + 1
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    iti      = 1000
    fixation = 1000
    M1       = 500
    M2       = 500
    M3       = 500
    T        = iti + fixation + M1 + M2 + M3

    epochs = {
        'iti':      (0, iti),
        'fixation': (iti, iti + fixation),
        'M1':       (iti + fixation, iti + fixation + M1),
        'M2':       (iti + fixation + M1, iti + fixation + M1 + M2),
        'M3':       (iti + fixation + M1 + M2, iti + fixation + M1 + M2 + M3),
        }
    epochs['T'] = T

    #---------------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {'seq': seq}

    #---------------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------------

    # Input matrix
    X = np.zeros((len(t), Nin))

    # Which sequence?
    X[:,SEQUENCE[seq-1]] = 1

    # Sequence
    sequence = sequences[seq]

    # Options
    X[e['fixation'],sequence[0]] = 1
    for I, J in zip([e['M1'], e['M2'], e['M3']],
                    [[sequence[0]] + options[sequence[0]],
                     [sequence[1]] + options[sequence[1]],
                     [sequence[2]] + options[sequence[2]]]):
        for j in J:
            X[I,j] = 1

    # Inputs
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.zeros((len(t), Nout)) # Mask matrix

        # Hold gaze
        Y[e['fixation'],:] = target_position(sequence[0])
        Y[e['M1'],:]       = target_position(sequence[1])
        Y[e['M2'],:]       = target_position(sequence[2])
        Y[e['M3'],:]       = target_position(sequence[3])

        # We don't constrain the intertrial interval
        M[e['fixation']+e['M1']+e['M2']+e['M3'],:] = 1

        # Output and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #---------------------------------------------------------------------------------

    return trial

min_error = 0.06

mode         = 'continuous'
n_validation = 100*nseq

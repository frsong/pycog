"""
Random dot motion task with arbitrary target orientation, saccade output.

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Orientation
#-----------------------------------------------------------------------------------------

N_MT = 8
preferred_orientations = np.arange(-180, 180, 360/N_MT)

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

T_IN     = range(0,      N_MT)
T_OUT    = range(N_MT,   2*N_MT)
MOTION   = range(2*N_MT, 3*N_MT)
FIXATION = range(3*N_MT, 3*N_MT+1)

Nin  = len(T_IN + T_OUT + MOTION + FIXATION)
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

Ne = len(EXC)
Ni = len(INH)

EXC_SENSORY = EXC[:Ne//2]
INH_SENSORY = INH[:Ni//2]
EXC_SACCADE = EXC[Ne//2:]
INH_SACCADE = INH[Ni//2:]

#-----------------------------------------------------------------------------------------
# Time constant
#-----------------------------------------------------------------------------------------

tau = 50

#-----------------------------------------------------------------------------------------
# Input connectivity
#-----------------------------------------------------------------------------------------

lambda1_rec = 1
"""
Cin = np.zeros((N, Nin))
Cin[EXC_SENSORY + INH_SENSORY,:] = 1

#-----------------------------------------------------------------------------------------
# Recurrent connectivity
#-----------------------------------------------------------------------------------------

Crec = np.zeros((N, N))
for i in EXC_SENSORY:
    Crec[i,EXC_SENSORY] = 1
    Crec[i,INH_SENSORY] = np.sum(Crec[i,EXC])/len(INH_SENSORY)
for i in EXC_SACCADE:
    Crec[i,EXC_SENSORY] = 1
    Crec[i,EXC_SACCADE] = 1
    Crec[i,INH_SACCADE] = np.sum(Crec[i,EXC])/len(INH_SACCADE)
for i in INH_SENSORY:
    Crec[i,EXC_SENSORY] = 1
    Crec[i,INH_SENSORY] = np.sum(Crec[i,EXC])/len(INH_SENSORY)
for i in INH_SACCADE:
    Crec[i,EXC_SENSORY] = 1
    Crec[i,EXC_SACCADE] = 1
    Crec[i,INH_SACCADE] = np.sum(Crec[i,EXC])/len(INH_SACCADE)
"""
#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
#Cout[:,EXC_SACCADE + INH_SACCADE] = 1
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_in  = 0.01**2
var_rec = 0.1**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

cohs        = [1, 2, 4, 8, 16]
in_outs     = [1, -1]
nconditions = len(cohs)*len(in_outs)

coh_min = min(cohs)
coh_max = max(cohs)

r_saccade = 0.5

SCALE = 3.2
def scale(coh):
    return (1 + SCALE*coh/100)/2

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------------

    if params['name'] in ['gradient', 'test']:
        coh    = params.get('coh',    rng.choice(cohs))    # Coherence
        in_out = params.get('in_out', rng.choice(in_outs)) # Into or out of RF
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % nconditions

        k0, k1 = tasktools.unravel_index(b, (len(cohs), len(in_outs)))
        coh    = cohs[k0]
        in_out = in_outs[k1]
    else:
        raise ValueError("Unknown trial type.")

    Tin  = rng.uniform(0, 360) # RF-in
    Tout = (Tin + 180) % 360   # RF-out

    #---------------------------------------------------------------------------------
    # Epochs
    #---------------------------------------------------------------------------------

    if False:
        epochs = {'T': 1000}
    else:
        fixation = tasktools.uniform(rng, dt, 200, 600)
        targets  = tasktools.uniform(rng, dt, 200, 600)
        stimulus = tasktools.truncated_exponential(rng, dt, 330, xmin=80, xmax=1500)
        decision = 500
        T        = fixation + targets + stimulus + decision

        epochs = {
            'fixation': (0, fixation + targets + stimulus),
            'targets':  (fixation, T),
            'stimulus': (fixation + targets, fixation + targets + stimulus),
            'decision': (fixation + targets + stimulus, T)
            }
        epochs['T'] = T

    #---------------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    # Correct choice
    if in_out > 0:
        Tchoice = Tin
    else:
        Tchoice = Tout

    # Trial info
    trial['info'] = {'coh': coh, 'in_out': in_out,
                     'Tin': Tin, 'Tout': Tout, 'Tchoice': Tchoice}

    #---------------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------------

    # Input matrix
    X = np.zeros((len(t), Nin))

    # Fixation
    for i in e['fixation']:
        X[i,FIXATION] = 1

    # Targets
    for i in e['targets']:
        X[i,T_IN]  = tasktools.vonMises(Tin,  preferred_orientations, b=0.3, 
                                        g=scale(coh_max))
        X[i,T_OUT] = tasktools.vonMises(Tout, preferred_orientations, b=0.3, 
                                        g=scale(coh_max))

    # Stimulus
    for i in e['stimulus']:
        X[i,MOTION] = tasktools.vonMises(Tchoice, preferred_orientations, b=0.3,
                                         g=scale(coh))

    # Inputs
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        # Output matrix
        Y = np.zeros((len(t), Nout))

        # Don't respond until go cue
        Y[e['fixation'],:] = 0, 0

        # After go cue
        s = tasktools.deg2rad(Tchoice)
        Y[e['decision'],:] = r_saccade*np.array([np.cos(s), np.sin(s)])

        # Outputs
        trial['outputs'] = Y

    #---------------------------------------------------------------------------------

    return trial

# Termination criterion
min_error = 0.05

# Validation dataset
n_validation = 100*(nconditions + 1)

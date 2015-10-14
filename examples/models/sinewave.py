"""
Generating a sine wave with no input
"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 0
N    = 50
Nout = 1

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

tau = 100
period = 8*tau # period of the sine wave

#-----------------------------------------------------------------------------------------
# Simulation parameter
#-----------------------------------------------------------------------------------------

var_rec = 0.05**2 # recurrent noise is critical for generalization

# Training of bias to the recurrent units is necesary for this task
train_bout = True
train_brec = True

def generate_trial(rng, dt, params):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------
    T        = (rng.uniform(2*period,3*period))//dt*dt
    epochs = {'T': T}

    #-------------------------------------------------------------------------------------
    # Trial info
    #-------------------------------------------------------------------------------------
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------
    trial['inputs'] = None
    
    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    Y = np.zeros((len(t), Nout)) # Output
    Y[:,0] = np.sin(2*np.pi*t/period) # Assuming one output

    # Outputs and mask
    trial['outputs'] = Y
    #-------------------------------------------------------------------------------------

    return trial

# Target error
min_error = 0.05

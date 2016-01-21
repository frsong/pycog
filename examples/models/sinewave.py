"""
Generate a sine wave with no input.

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

N    = 50
Nout = 1

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant and step size (latter should be small for this example)
tau = 100
dt  = 5

# Biases are helpful for this task
train_bout = True
train_brec = True

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

# Period of the sine wave
period = 8*tau

def generate_trial(rng, dt, params):
    # Sample duration
    epochs = {'T': 2*period}

    # Trial info
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    # Target output
    trial['outputs'] = 0.9*np.sin(2*np.pi*t/period)[:,None]

    return trial

# Target error
min_error = 0.02

# Online training
mode         = 'continuous'
n_validation = 100

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    from pycog import Model

    model = Model(N=N, Nout=Nout, ei=ei, tau=tau, dt=dt,
                  train_brec=train_brec, train_bout=train_bout, var_rec=var_rec,
                  generate_trial=generate_trial,
                  mode=mode, n_validation=n_validation, min_error=min_error)
    model.train('savefile.pkl')

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    from pycog          import RNN
    from pycog.figtools import Figure

    rnn  = RNN('savefile.pkl', {'dt': 0.5})
    info = rnn.run(T=16*period)

    fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.z[0], color=Figure.colors('blue'))
    plot.ylim(-1, 1)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel('$\sin t$')

    fig.save(path='.', name='sinewave')

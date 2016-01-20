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

# Time constant
tau = 100

# Biases are really helpful for this task
train_bout = True
train_brec = True

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.05**2 # Helps with generalization

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

# Period of the sine wave
period = 8*tau

def generate_trial(rng, dt, params):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    T      = (rng.uniform(2*period, 3*period))//dt*dt
    epochs = {'T': T}

    #-------------------------------------------------------------------------------------
    # Trial info
    #-------------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    Y      = np.zeros((len(t), Nout)) # Output
    Y[:,0] = np.sin(2*np.pi*t/period) # Assuming one output

    # Set output
    trial['outputs'] = Y

    #-------------------------------------------------------------------------------------

    return trial

# Target error
min_error = 0.05

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    from pycog import Model

    model = Model(N=N, Nout=Nout, ei=ei, tau=tau,
                  train_brec=train_brec, train_bout=train_bout,
                  var_rec=var_rec, generate_trial=generate_trial, min_error=min_error)
    model.train('savefile.pkl')

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    from pycog          import RNN
    from pycog.figtools import Figure

    rnn  = RNN('savefile.pkl', {'dt': 0.5})
    info = rnn.run(T=10*period)

    fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.z[0], color=Figure.colors('blue'))
    plot.highlight(2*period/tau, 3*period/tau)
    plot.text_upper_right('Range of training durations highlighted', dy=0.02)

    plot.ylim(-1, 1)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel('$\sin t$')

    fig.save(path='.', name='sinewave')

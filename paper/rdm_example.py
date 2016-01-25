"""
Perceptual decision-making task, loosely based on the random dot motion task.

"""
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
left_rights = [1, -1]
nconditions = len(cohs)*len(left_rights)
pcatch      = 1/(nconditions + 1)

SCALE = 3.2
def scale(coh):
    return (1 + SCALE*coh/100)/2

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------

    if params.get('catch', rng.rand() < pcatch):
        catch_trial = True
    else:
        catch_trial = False
        coh         = params.get('coh',        rng.choice(cohs))
        left_right  = params.get('left_right', rng.choice(left_rights))

    #---------------------------------------------------------------------------
    # Epochs
    #---------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 2000}
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

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        if left_right > 0:
            choice = 0
        else:
            choice = 1

        # Trial info
        trial['info'] = {'coh': coh, 'left_right': left_right, 'choice': choice}

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

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    # Train model
    model = Model(Nin=Nin, N=N, Nout=Nout, ei=ei, Cout=Cout,
                  generate_trial=generate_trial,
                  performance=performance, terminate=terminate,
                  n_validation=n_validation)
    model.train('savefile.pkl')

    # Run the trained network with 16*3.2% = 51.2% coherence for choice 1
    rnn        = RNN('savefile.pkl', {'dt': 0.5})
    trial_func = generate_trial
    trial_args = {'name': 'test', 'catch': False, 'coh': 16, 'left_right': 1}
    info       = rnn.run(inputs=(trial_func, trial_args))

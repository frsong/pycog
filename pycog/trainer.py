"""
Train a recurrent neural network.

"""
from __future__ import absolute_import
from __future__ import division

import os
import sys
from   collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

from .             import theanotools
from .connectivity import Connectivity
from .dataset      import Dataset
from .defaults     import defaults
from .rnn          import RNN
from .sgd          import SGD
from .utils        import print_settings

THIS = 'pycog.trainer'

class Trainer(object):
    """
    Train an RNN.

    """
    def __init__(self, params, floatX=theano.config.floatX):
        """
        Initialize.

        Parameters
        ----------

        params : dict
                 All parameters have default values, see `pycog.defaults`.

          Entries
          -------

          Nin : int, optional
                Number of input units.

          N : int, optional
              Number of recurrent units.

          Nout : int, optional
                 Number of output units.

          train_brec : bool, optional
                       Whether to train recurrent biases.

          brec : float, optional
                 Initial value of the recurrent bias.

          train_bout : bool, optional
                       Whether to train output biases.

          bout : float, optional
                 Initial value of the output bias

          train_x0 : bool, optional
                     Whether to optimize the initial conditions.

          x0 : float, optional
               Initial value of the initial conditions.

          mode : str, optional
                 `continuous` or `batch` training/running mode.

          tau : float or numpy.ndarray, optional
                Time constant(s) for recurrent units.

          Cin : numpy.ndarray or Connectivity, optional
                Input weight structure.

          Crec : numpy.ndarray or Connectivity, optional
                 Recurrent weight structure.

          Cout : numpy.ndarray or Connectivity, optional
                 Output weight structure.

          ei : numpy.ndarray, optional
               E/I signature.

          ei_positive_func : str, optional
                             Function to use to keep weights positive.

          hidden_activation : str, optional
                              Hidden activation function.

          output_activation: str, optional
                             Output activation function.

          n_gradient : int, optional
                       Minibatch size for gradient dataset.

          n_validation : int, optional
                         Minibatch size for validation dataset.

          gradient_batch_size, validation_batch_size : int, optional
                                                       Number of trials to precompute
                                                       and store in each dataset. Make
                                                       sure the batch sizes are larger
                                                       than the minibatch sizes.

          lambda_Omega : float, optinonal
                         Multiplier for the vanishing gradient regularizer.

          lambda1_in, lambda1_rec, lambda1_out : float, optional
                                                 Multipliers for L1 weight regularization.

          lambda2_in, lambda2_rec, lambda2_out : float, optional
                                                 Multipliers for L2 weight regularization.

          lambda2_r : float, optional
                      Multiplier for L2 firing rate regularization.

          callback : function, optional
                     Evaluate validation dataset.

          performance : function, optional
                        Performance measure.

          terminate : function, optional
                      Custom termination criterion.

          min_error : float, optional
                      Target error. Terminate if error is less than or equal.

          learning_rate : float, optional
                          Learning rate for gradient descent.

          max_gradient_norm : float, optional
                              Clip gradient if its norm is greater than.

          bound : float, optional
                  Lower bound for denominator in vanishing gradient regularizer.

          baseline_in : float, optional
                        Baseline input rate.

          var_in : float or numpy.ndarray, optional
                   Variance(s) for inputs.

          var_rec : float or numpy.ndarray, optional
                    If `float` or 1D `numpy.ndarray`, then recurrent units receive
                    independent noise. If 2D `numpy.ndarray`, then noise is drawn
                    from a multivariable normal distribution.

          seed, gradient_seed, validation_seed : int, optional
                                                 Seeds for the random number generators.

          structure : dict, optional
                      Convey structure information, such as what each input represents.

          rho0 : float, optional
                 Spectral radius for the initial recurrent weight matrix.

          max_iter : int, optional
                     Maximum number of iterations for gradient descent.

          dt : float, optional
               Integration time step.

          distribution_in : str, optional
                            Distribution for the initial input weight matrix.

          distribution_rec : str, optional
                             Distribution for the initial recurrent weight matrix.

          distribution_out : str, optional
                             Distribution for the initial output weight matrix.

          gamma_k : float, optional
                    k in Gamma(k, theta). Note mean = k*theta, var = k*theta^2.

          checkfreq : int, optional
                      Frequency with which to evaluate validation error.

          patience : int, optional
                     Terminate training if the objective function doesn't change
                     for longer than `patience`.

          Not used currently
          ------------------

          momentum, method

        floatX : str, optional
                 Floating-point type.

        """
        self.p      = params.copy()
        self.floatX = floatX

        #---------------------------------------------------------------------------------
        # Fill in default parameters
        #---------------------------------------------------------------------------------

        # Default parameters
        for k in defaults:
            self.p.setdefault(k, defaults[k])

        # Time constants
        if not np.isscalar(self.p['tau']):
            self.p['tau'] = np.asarray(self.p['tau'], dtype=floatX)

        # Time step
        if self.p['dt'] is None:
            if np.isscalar(self.p['tau']):
                self.p['dt'] = self.p['tau']/5
            else:
                self.p['dt'] = np.min(self.p['tau'])/5

        # Distribution for initial weights (Win)
        if self.p['distribution_in'] is None:
            if self.p['ei'] is not None:
                self.p['distribution_in'] = 'uniform'
            else:
                self.p['distribution_in'] = 'uniform'

        # Distribution for initial weights (Wrec)
        if self.p['distribution_rec'] is None:
            if self.p['ei'] is not None:
                self.p['distribution_rec'] = 'gamma'
            else:
                self.p['distribution_rec'] = 'normal'

        # Distribution for initial weights (Wout)
        if self.p['distribution_out'] is None:
            if self.p['ei'] is not None:
                self.p['distribution_out'] = 'uniform'
            else:
                self.p['distribution_out'] = 'uniform'

        # Default mask for recurrent weights
        if self.p['Crec'] is None:
            N = self.p['N']
            if self.p['ei'] is not None:
                # Default for E/I is fully (non-self) connected, mean-balanced
                exc, = np.where(self.p['ei'] > 0)
                inh, = np.where(self.p['ei'] < 0)

                C = np.zeros((N, N))
                for i in exc:
                    C[i,exc]  = 1
                    C[i,i]    = 0
                    C[i,inh]  = 1
                    C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
                for i in inh:
                    C[i,exc]  = 1
                    C[i,inh]  = 1
                    C[i,i]    = 0
                    C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
                C /= np.linalg.norm(C, axis=1)[:,np.newaxis]

                self.p['Crec'] = C
            else:
                # Default for no E/I is fully (non-self) connected
                C = np.ones((N, N))
                np.fill_diagonal(C, 0)

                self.p['Crec'] = C

        # Convert to connectivity matrices
        for k in ['Cin', 'Crec', 'Cout']:
            if self.p[k] is not None and not isinstance(self.p[k], Connectivity):
                self.p[k] = Connectivity(self.p[k])

    #/////////////////////////////////////////////////////////////////////////////////////

    def init_weights(self, rng, C, m, n, distribution):
        """
        Initialize weights from a distribution.

        Parameters
        ----------

        rng : numpy.random.RandomState
              Random number generator.

        C : Connectivity
            Specify which weights are plastic and nonzero.

        m, n : int
               Number of rows and columns, respectively.

        distribution : str
                       Name of the distribution.

        """
        # Account for plastic and fixed weights.
        if C is not None:
            mask = C.plastic
            size = C.nplastic
        else:
            mask = 1
            size = m*n

        # Distributions
        if distribution == 'uniform':
            w = 0.1*rng.uniform(-mask, mask, size=size)
        elif distribution == 'normal':
            w = rng.normal(np.zeros(size), mask, size=size)
        elif distribution == 'gamma':
            k     = self.p['gamma_k']
            theta = 0.1*mask/k
            w     = rng.gamma(k, theta, size=size)
        elif distribution == 'lognormal':
            mean  = 0.5*mask
            var   = 0.1
            mu    = np.log(mean/np.sqrt(1 + var/mean**2))
            sigma = np.sqrt(np.log(1 + var/mean**2))
            w     = rng.lognormal(mu, sigma, size=size)
        else:
            raise NotImplementedError("[ {}.Trainer.init_weights ] distribution: {}"
                                      .format(THIS, distribution))

        if C is not None:
            W = np.zeros(m*n)
            W[C.idx_plastic] = w
        else:
            W = w

        return W.reshape((m, n))

    def train(self, savefile, task, recover=True):
        """
        Train the RNN.

        Parameters
        ----------

        savefile : str

        task : function

        recover : bool, optional
                  If `True`, will attempt to recover from a previously saved run.

        """
        N     = self.p['N']
        Nin   = self.p['Nin']
        Nout  = self.p['Nout']
        alpha = self.p['dt']/self.p['tau']

        # Initialize settings
        settings = OrderedDict()

        # Check if file already exists
        if not recover:
            if os.path.isfile(savefile):
                os.remove(savefile)

        #---------------------------------------------------------------------------------
        # Are we using GPUs?
        #---------------------------------------------------------------------------------

        if theanotools.get_processor_type() == 'gpu':
            settings['GPU'] = 'enabled'
        else:
            settings['GPU'] = 'no'

        #---------------------------------------------------------------------------------
        # Random number generator
        #---------------------------------------------------------------------------------

        settings['init seed'] = self.p['seed']
        rng = np.random.RandomState(self.p['seed'])

        #---------------------------------------------------------------------------------
        # Weight initialization
        #---------------------------------------------------------------------------------

        settings['distribution (Win)']  = self.p['distribution_in']
        settings['distribution (Wrec)'] = self.p['distribution_rec']
        settings['distribution (Wout)'] = self.p['distribution_out']

        if Nin > 0:
            Win_0 = self.init_weights(rng, self.p['Cin'], N, Nin,
                                      self.p['distribution_in'])
        Wrec_0 = self.init_weights(rng, self.p['Crec'],
                                   N, N, self.p['distribution_rec'])
        Wout_0 = self.init_weights(rng, self.p['Cout'],
                                   Nout, N, self.p['distribution_out'])

        #---------------------------------------------------------------------------------
        # Enforce Dale's law on the initial weights
        #---------------------------------------------------------------------------------

        settings['Nin/N/Nout'] = '{}/{}/{}'.format(Nin, N, Nout)

        if self.p['ei'] is not None:
            Nexc = len(np.where(self.p['ei'] > 0)[0])
            Ninh = len(np.where(self.p['ei'] < 0)[0])
            settings['Dale\'s law'] = 'E/I = {}/{}'.format(Nexc, Ninh)

            if Nin > 0:
                Win_0 = abs(Win_0) # If Dale, assume inputs are excitatory
            Wrec_0 = abs(Wrec_0)
            Wout_0 = abs(Wout_0)
        else:
            settings['Dale\'s law'] = 'no'

        #---------------------------------------------------------------------------------
        # Fix spectral radius
        #---------------------------------------------------------------------------------

        # Compute spectral radius
        C = self.p['Crec']
        if C is not None:
            Wrec_0_full = C.mask_plastic*Wrec_0 + C.mask_fixed
        else:
            Wrec_0_full = Wrec_0
        if self.p['ei'] is not None:
            Wrec_0_full = Wrec_0_full*self.p['ei']
        rho = RNN.spectral_radius(Wrec_0_full)

        # Scale Wrec to have fixed spectral radius
        if self.p['ei'] is not None:
            R = self.p['rho0']/rho
        else:
            R = 1.1/rho
        Wrec_0 *= R
        if C is not None:
            C.mask_fixed *= R

        # Check spectral radius
        if C is not None:
            Wrec_0_full = C.mask_plastic*Wrec_0 + C.mask_fixed
        else:
            Wrec_0_full = Wrec_0
        if self.p['ei'] is not None:
            Wrec_0_full = Wrec_0_full*self.p['ei']
        rho = RNN.spectral_radius(Wrec_0_full)
        settings['initial spectral radius'] = '{:.2f}'.format(rho)

        #---------------------------------------------------------------------------------
        # Others
        #---------------------------------------------------------------------------------

        brec_0 = self.p['brec']*np.ones(N)
        bout_0 = self.p['bout']*np.ones(Nout)
        x0_0   = self.p['x0']*np.ones(N)

        #---------------------------------------------------------------------------------
        # RNN parameters
        #---------------------------------------------------------------------------------

        if Nin > 0:
            Win = theanotools.shared(Win_0, name='Win')
        else:
            Win = None
        Wrec = theanotools.shared(Wrec_0, name='Wrec')
        Wout = theanotools.shared(Wout_0, name='Wout')
        brec = theanotools.shared(brec_0, name='brec')
        bout = theanotools.shared(bout_0, name='bout')
        x0   = theanotools.shared(x0_0,   name='x0')

        #---------------------------------------------------------------------------------
        # Parameters to train
        #---------------------------------------------------------------------------------

        trainables = []
        if Win is not None:
            trainables += [Win]
        trainables += [Wrec]
        if Wout is not None:
            trainables += [Wout]

        if self.p['train_brec']:
            settings['train recurrent bias'] = 'yes'
            trainables += [brec]
        else:
            settings['train recurrent bias'] = 'no'

        if self.p['train_bout']:
            settings['train output bias'] = 'yes'
            trainables += [bout]
        else:
            settings['train output bias'] = 'no'

        # In continuous mode it doesn't make sense to train x0, which is forgotten
        if self.p['mode'] == 'continuous':
            self.p['train_x0'] = False

        if self.p['train_x0']:
            settings['train initial conditions'] = 'yes'
            trainables += [x0]
        else:
            settings['train initial conditions'] = 'no'

        #---------------------------------------------------------------------------------
        # Weight matrices
        #---------------------------------------------------------------------------------

        # Input
        if Nin > 0:
            if self.p['Cin'] is not None:
                C = self.p['Cin']
                settings['sparseness (Win)'] = ('p = {:.2f}, p_plastic = {:.2f}'
                                                .format(C.p, C.p_plastic))

                Cin_mask_plastic = theanotools.shared(C.mask_plastic)
                Cin_mask_fixed   = theanotools.shared(C.mask_fixed)

                Win_ = Cin_mask_plastic*Win + Cin_mask_fixed
                Win_.name = 'Win_'
            else:
                Win_ = Win

        # Recurrent
        if self.p['Crec'] is not None:
            C = self.p['Crec']
            settings['sparseness (Wrec)'] = ('p = {:.2f}, p_plastic = {:.2f}'
                                             .format(C.p, C.p_plastic))

            Crec_mask_plastic = theanotools.shared(C.mask_plastic)
            Crec_mask_fixed   = theanotools.shared(C.mask_fixed)

            Wrec_ = Crec_mask_plastic*Wrec + Crec_mask_fixed
            Wrec_.name = 'Wrec_'
        else:
            Wrec_ = Wrec

        # Output
        if self.p['Cout'] is not None:
            C = self.p['Cout']
            settings['sparseness (Wout)'] = ('p = {:.2f}, p_plastic = {:.2f}'
                                             .format(C.p, C.p_plastic))

            Cout_mask_plastic = theanotools.shared(C.mask_plastic)
            Cout_mask_fixed   = theanotools.shared(C.mask_fixed)

            Wout_ = Cout_mask_plastic*Wout + Cout_mask_fixed
            Wout_.name = 'Wout_'
        else:
            Wout_ = Wout

        #---------------------------------------------------------------------------------
        # Dale's law
        #---------------------------------------------------------------------------------

        if self.p['ei'] is not None:
            # Function to keep matrix elements positive
            if self.p['ei_positive_func'] == 'abs':
                settings['E/I positivity function'] = 'absolute value'
                make_positive = abs
            elif self.p['ei_positive_func'] == 'rectify':
                settings['E/I positivity function'] = 'rectify'
                make_positive = theanotools.rectify
            else:
                raise ValueError("Unknown ei_positive_func.")

            # Assume inputs are excitatory
            if Nin > 0:
                Win_ = make_positive(Win_)

            # E/I
            ei    = theanotools.shared(self.p['ei'], name='ei')
            Wrec_ = make_positive(Wrec_)*ei

            Wout_ = make_positive(Wout_)*ei

        #---------------------------------------------------------------------------------
        # Variables to save
        #---------------------------------------------------------------------------------

        if Nin > 0:
            save_values = [Win_]
        else:
            save_values = [None]
        save_values += [Wrec_, Wout_, brec, bout, x0]

        #---------------------------------------------------------------------------------
        # Activation functions
        #---------------------------------------------------------------------------------

        f_hidden, d_f_hidden = theanotools.hidden_activations[self.p['hidden_activation']]
        settings['hidden activation'] = self.p['hidden_activation']

        act = self.p['output_activation']
        f_output = theanotools.output_activations[act]

        if act == 'sigmoid':
            settings['output activation/loss'] = 'sigmoid/binary cross entropy'
            f_loss = theanotools.binary_crossentropy
        elif act == 'softmax':
            settings['output activation/loss'] = 'softmax/categorical cross entropy'
            f_loss = theanotools.categorical_crossentropy
        else:
            settings['output activation/loss'] = act + '/squared'
            f_loss = theanotools.L2

        #---------------------------------------------------------------------------------
        # RNN
        #---------------------------------------------------------------------------------

        # Dims: time, trials, units
        # u[:,:,:Nin]  contains the inputs (including baseline and noise),
        # u[:,:,Nin:]  contains the recurrent noise
        u   = T.tensor3('u')
        x0_ = T.alloc(x0, u.shape[1], x0.shape[0])

        if Nin > 0:
            def rnn(u_t, x_tm1, r_tm1, WinT, WrecT):
                x_t = ((1 - alpha)*x_tm1
                       + alpha*(T.dot(r_tm1, WrecT)        # Recurrent
                                + brec                     # Bias
                                + T.dot(u_t[:,:Nin], WinT) # Input
                                + u_t[:,Nin:])             # Recurrent noise
                       )
                r_t = f_hidden(x_t)

                return [x_t, r_t]

            [x, r], _ = theano.scan(fn=rnn,
                                    outputs_info=[x0_, f_hidden(x0_)],
                                    sequences=u,
                                    non_sequences=[Win_.T, Wrec_.T])
        else:
            def rnn(u_t, x_tm1, r_tm1, WrecT):
                x_t = ((1 - alpha)*x_tm1
                       + alpha*(T.dot(r_tm1, WrecT) # Recurrent
                                + brec              # Bias
                                + u_t[:,Nin:]) # Recurrent noise
                       )
                r_t = f_hidden(x_t)

                return [x_t, r_t]

            [x, r], _ = theano.scan(fn=rnn,
                                    outputs_info=[x0_, f_hidden(x0_)],
                                    sequences=u,
                                    non_sequences=[Wrec_.T])

        #---------------------------------------------------------------------------------
        # Running mode
        #---------------------------------------------------------------------------------

        if self.p['mode'] == 'continuous':
            settings['mode'] = 'continuous'

            if self.p['n_gradient'] != 1:
                print("[ Trainer.train ] In continuous mode,"
                      " so we're setting n_gradient to 1.")
                self.p['n_gradient'] = 1

            x0_ = x[-1]
        else:
            settings['mode'] = 'batch'

        #---------------------------------------------------------------------------------
        # Readout
        #---------------------------------------------------------------------------------

        z = f_output(T.dot(r, Wout_.T) + bout)

        #---------------------------------------------------------------------------------
        # Deduce whether the task specification contains an output mask -- use a
        # temporary dataset so it doesn't affect the training.
        #---------------------------------------------------------------------------------

        dataset = Dataset(1, task, self.floatX, self.p, name='gradient')
        if dataset.has_output_mask():
            settings['output mask'] = 'yes'
        else:
            settings['output mask'] = 'no'

        #---------------------------------------------------------------------------------
        # Loss
        #---------------------------------------------------------------------------------

        # (time, trials, outputs)
        target = T.tensor3('target')

        # Set mask
        mask     = target[:,:,Nout:]
        masknorm = T.sum(mask)

        # Input-output pairs
        inputs = [u, target]
        # target[:,:,:Nout] contains the target outputs, &
        # target[:,:,Nout:] contains the mask.

        # Loss, not including the regularization terms
        loss = T.sum(f_loss(z, target[:,:,:Nout])*mask)/masknorm

        # Root-mean-squared error
        error = T.sqrt(T.sum(theanotools.L2(z, target[:,:,:Nout])*mask)/masknorm)

        #---------------------------------------------------------------------------------
        # Regularization terms
        #---------------------------------------------------------------------------------

        regs = 0

        #---------------------------------------------------------------------------------
        # L1 weight regularization
        #---------------------------------------------------------------------------------

        lambda1 = self.p['lambda1_in']
        if lambda1 > 0:
            settings['L1 weight regularization (Win)'] = ('lambda1_in = {}'
                                                          .format(lambda1))
            regs += lambda1 * T.mean(abs(Win))

        lambda1 = self.p['lambda1_rec']
        if lambda1 > 0:
            settings['L1 weight regularization (Wrec)'] = ('lambda1_rec = {}'
                                                           .format(lambda1))
            regs += lambda1 * T.mean(abs(Wrec))

        lambda1 = self.p['lambda1_out']
        if lambda1 > 0:
            settings['L1 weight regularization (Wout)'] = ('lambda1_out = {}'
                                                           .format(lambda1))
            regs += lambda1 * T.mean(abs(Wout))

        #---------------------------------------------------------------------------------
        # L2 weight regularization
        #---------------------------------------------------------------------------------

        if Nin > 0:
            lambda2 = self.p['lambda2_in']
            if lambda2 > 0:
                settings['L2 weight regularization (Win)'] = ('lambda2_in = {}'
                                                              .format(lambda2))
                regs += lambda2 * T.mean(Win**2)

        lambda2 = self.p['lambda2_rec']
        if lambda2 > 0:
            settings['L2 weight regularization (Wrec)'] = ('lambda2_rec = {}'
                                                           .format(lambda2))
            regs += lambda2 * T.mean(Wrec**2)

        lambda2 = self.p['lambda2_out']
        if lambda2 > 0:
            settings['L2 weight regularization (Wout)'] = ('lambda2_out = {}'
                                                           .format(lambda2))
            regs += lambda2 * T.mean(Wout**2)

        #---------------------------------------------------------------------------------
        # L2 rate regularization
        #---------------------------------------------------------------------------------

        lambda2 = self.p['lambda2_r']
        if lambda2 > 0:
            settings['L2 rate regularization'] = 'lambda2_r = {}'.format(lambda2)
            regs += lambda2 * T.mean(r**2)

        #---------------------------------------------------------------------------------
        # Final costs
        #---------------------------------------------------------------------------------

        costs = [loss, error]

        #---------------------------------------------------------------------------------
        # Datasets
        #---------------------------------------------------------------------------------

        gradient_data   = Dataset(self.p['n_gradient'], task, self.floatX, self.p,
                                  batch_size=self.p['gradient_batch_size'],
                                  seed=self.p['gradient_seed'],
                                  name='gradient')
        validation_data = Dataset(self.p['n_validation'], task, self.floatX, self.p,
                                  batch_size=self.p['validation_batch_size'],
                                  seed=self.p['validation_seed'],
                                  name='validation')

        # Input noise
        if np.isscalar(self.p['var_in']):
            if Nin > 0:
                settings['sigma_in'] = '{}'.format(np.sqrt(self.p['var_in']))
        else:
            settings['sigma_in'] = 'array'

        # Recurrent noise
        if np.isscalar(self.p['var_rec']):
            settings['sigma_rec'] = '{}'.format(np.sqrt(self.p['var_rec']))
        else:
            settings['sigma_rec'] = 'array'

        # Dataset settings
        settings['rectify inputs']            = self.p['rectify_inputs']
        settings['gradient minibatch size']   = gradient_data.minibatch_size
        settings['validation minibatch size'] = validation_data.minibatch_size

        #---------------------------------------------------------------------------------
        # Other settings
        #---------------------------------------------------------------------------------

        settings['dt'] = '{} ms'.format(self.p['dt'])
        if np.isscalar(self.p['tau']):
            settings['tau'] = '{} ms'.format(self.p['tau'])
        else:
            settings['tau'] = 'custom'
        settings['tau_in']            = '{} ms'.format(self.p['tau_in'])
        settings['learning rate']     = '{}'.format(self.p['learning_rate'])
        settings['lambda_Omega']      = '{}'.format(self.p['lambda_Omega'])
        settings['max gradient norm'] = '{}'.format(self.p['max_gradient_norm'])

        #---------------------------------------------------------------------------------
        # A few important Theano settings
        #---------------------------------------------------------------------------------

        settings['(Theano) floatX']   = self.floatX
        settings['(Theano) allow_gc'] = theano.config.allow_gc

        #---------------------------------------------------------------------------------
        # Train!
        #---------------------------------------------------------------------------------

        print_settings(settings)

        sgd = SGD(trainables, inputs, costs, regs, x, z, self.p, save_values,
                  {'Wrec_': Wrec_, 'd_f_hidden': d_f_hidden})
        sgd.train(gradient_data, validation_data, savefile)

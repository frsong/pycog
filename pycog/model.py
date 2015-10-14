"""
Wrapper class for training RNNs.

"""
from __future__ import absolute_import

import imp
import inspect
import os
import sys

import numpy as np

from .defaults import defaults

THIS = 'pycog.model'

class Struct():
    """
    Treat a dictionary like a module.

    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Model(object):
    """
    Provide a simpler interface to users, and check for obvious mistakes.

    """
    def __init__(self, modelfile=None, **kwargs):
        """
        If `modelfile` is provided infer everything from the file, otherwise the
        user is responsible for providing the necessary parameters through `kwargs`.

        Parameters
        ----------

        modelfile: str
                   A Python script containing model parameters.

        kwargs : dict

          Should contain

          generate_trial : function
                           Return a dictionary containing trial information.
                           This function should take as its arguments

                           rng    : numpy.random.RandomState
                           dt     : float
                           params : dict

        """
        if modelfile is None:
            import sinewave
            print("[ {}.Model ] No modelfile provided. Train a sine wave.".format(THIS))
            self.m = sinewave
        else:
            try:
                self.m = imp.load_source('model', modelfile)
            except IOError:
                print("[ {}.Model ] Couldn't load model file {}".format(THIS, modelfile))
                sys.exit(1)

        #---------------------------------------------------------------------------------
        # Perform model check
        #---------------------------------------------------------------------------------
        # generate_trial : existence
        try:
            f    = self.m.generate_trial
            args = inspect.getargspec(f).args
        except AttributeError:
            print("[ {}.Model ] No generate_trial function defined. Generate a sine wave."
                  .format(THIS))
            import sinewave
            self.m.generate_trial = sinewave.generate_trial

        # generate_trial : usage
        if args != ['rng', 'dt', 'params']:
            print(("[ {}.Model ] Warning: Function generate_trial doesn't have the"
                   " expected list of argument names. It is OK if only the names are"
                   " different.").format(THIS))

        # var_in : If var_in is an array, make sure it is the correct size
        if (hasattr(self.m, 'var_in') and isinstance(self.m.var_in, np.ndarray)
            and self.m.var_in.ndim == 1):
            if len(self.m.var_in) != self.m.Nin:
                print("[ {}.Model ] The length of var_in doesn't match Nin.".format(THIS))
                sys.exit(1)

    #/////////////////////////////////////////////////////////////////////////////////////

    def train(self, savefile, seed=None, compiledir=None, recover=True, gpus=0):
        """
        Train the network.

        Parameters
        ----------

        savefile : str
        seed : int, optional
        compiledir : str, optional
        recover : bool, optional
        gpus : int, optional

        """
        # Theano setup
        os.environ.setdefault('THEANO_FLAGS', '')
        if compiledir is not None:
            os.environ['THEANO_FLAGS'] += ',base_compiledir=' + compiledir
        os.environ['THEANO_FLAGS'] += ',floatX=float32,allow_gc=False'
        if gpus > 0:
            os.environ['THEANO_FLAGS'] += ',device=gpu,nvcc.fastmath=True'

        # Only involve Theano for training
        from .trainer import Trainer

        # The task
        #TODO: Replace task directly with generate_trial?
        task = Struct(generate_trial=self.m.generate_trial)

        # Parameters
        params = {}

        # Seed, if given
        if seed is not None:
            params['seed'] = seed

        # Optional parameters
        for k in defaults:
            if hasattr(self.m, k):
                params[k] = getattr(self.m, k)

        # Train
        trainer = Trainer(params)
        trainer.train(savefile, task, recover=recover)

from __future__ import division

import numpy as np

class Dataset(object):
    """
    Dataset for training.

    """
    @staticmethod
    def rectify(x):
        return x*(x > 0)

    #/////////////////////////////////////////////////////////////////////////////////////

    def __init__(self, size, task, floatX, p, batch_size=None, seed=1, name='Dataset'):
        """

        Parameters
        ----------

        size : int
               Number of trials in each minibatch.

        task : Python function

        floatX : dtype
                 Floating-point type for NumPy arrays.

        p : dict
            Parameters.

        batch_size : int, optional
                     Number of trials to store. If `None`, same as `size`.

        seed : int
               Seed for random number generator.

        name : str
               Name of the dataset, which can be used by `task`, e.g., to distinguish
               between gradient and validation datasets.

        """
        self.minibatch_size = size
        self.task           = task
        self.floatX         = floatX
        self.name           = name
        self.network_shape  = (p['Nin'], p['N'], p['Nout'])
        for k in ['dt', 'rectify_inputs', 'baseline_in']:
            setattr(self, k, p[k])

        # Rescale noise
        self.var_in  = 2*p['tau']/p['dt']*p['var_in']
        self.var_rec = 2*p['tau']/p['dt']*p['var_rec']

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # Batch size
        if batch_size is None:
            batch_size = size
        self.batch_size = batch_size

        # Initialize
        self.inputs    = None
        self.targets   = None
        self.trials    = []
        self.ntrials   = 0
        self.trial_idx = self.batch_size

    #/////////////////////////////////////////////////////////////////////////////////////

    def has_output_mask(self):
        """
        Determine whether the outputs have a time mask.

        """
        # Generate a trial
        params = {
            'target_output': True,
            'name':          self.name
            }
        trial = self.task.generate_trial(self.rng, self.dt, params)

        return ('mask' in trial)

    #/////////////////////////////////////////////////////////////////////////////////////

    def __call__(self, best_costs, update=True):
        """
        Return a batch of trials.

        best_costs : list
                     The best costs, not including the loss.

        update : bool
                 If `True`, return a new set of trials.

        """
        if update:
            self.update(best_costs)
            self.ntrials += self.minibatch_size

        return [self.inputs [:,self.trial_idx:self.trial_idx+self.minibatch_size,:],
                self.targets[:,self.trial_idx:self.trial_idx+self.minibatch_size,:]]

    def get_trials(self):
        """
        Return info for current trials.

        """
        return self.trials[self.trial_idx:self.trial_idx+self.minibatch_size]

    def update(self, best_costs):
        """
        Generate a new minibatch of trials and store them in `self.inputs` and
        `self.outputs`. For speed (but at the cost of memory), we store `batch_size`
        trials and only create new trials if we run out of trials.

        For both inputs and outputs, the first two dimensions contain time and
        trials, respectively. For the third dimension,

          self.inputs [:,:,:Nin]  contains the inputs (including baseline and noise),
          self.inputs [:,:,Nin:]  contains the recurrent noise,
          self.outputs[:,:,:Nout] contains the target outputs, &
          self.outputs[:,:,Nout:] contains the mask.

        Parameters
        ----------

        best_costs : list of floats
                     Performance measures, in case you want to modify the trials
                     (e.g., noise) depending on the error.

        """
        self.trial_idx += self.minibatch_size
        if self.trial_idx + self.minibatch_size > self.batch_size:
            self.trial_idx = 0

            self.trials = []
            for b in xrange(self.batch_size):
                params = {
                    'target_output':   True,
                    'minibatch_index': b,
                    'best_costs':      best_costs,
                    'name':            self.name
                    }
                self.trials.append(self.task.generate_trial(self.rng, self.dt, params))

            # Longest trial
            k = np.argmax([len(trial['t']) for trial in self.trials])
            t = self.trials[k]['t']

            # Input and output matrices
            Nin, N, Nout = self.network_shape
            T = len(t)
            B = self.batch_size
            x = np.zeros((T, B, Nin+N),  dtype=self.floatX)
            y = np.zeros((T, B, 2*Nout), dtype=self.floatX)

            # Pad trials
            for b, trial in enumerate(self.trials):
                Nt = len(trial['t'])
                x[:Nt,b,:Nin]  = trial['inputs']
                y[:Nt,b,:Nout] = trial['outputs']
                if 'mask' in trial:
                    y[:Nt,b,Nout:] = trial['mask']
                else:
                    y[:Nt,b,Nout:] = 1

            # Input noise
            if Nin > 0:
                if np.isscalar(self.var_in) or self.var_in.ndim == 1:
                    # Independent noise
                    if np.any(self.var_in > 0):
                        r = np.sqrt(self.var_in)*self.rng.normal(size=(T, B, Nin))
                    else:
                        r = 0
                else:
                    # Correlated noise
                    r = self.rng.multivariate_normal(np.zeros(Nin), self.var_in, (T, B))
                x[:,:,:Nin] += self.baseline_in + r

            # Recurrent noise
            if np.isscalar(self.var_rec) or self.var_rec.ndim == 1:
                # Independent noise
                if np.any(self.var_rec > 0):
                    r = np.sqrt(self.var_rec)*self.rng.normal(size=(T, B, N))
                else:
                    r = 0
            else:
                # Correlated noise
                r = self.rng.multivariate_normal(np.zeros(N), self.var_rec, (T, B))
            x[:,:,Nin:] = r

            # Keep inputs positive
            if self.rectify_inputs:
                x[:,:,:Nin] = Dataset.rectify(x[:,:,:Nin])

            # Save
            self.inputs  = x
            self.targets = y

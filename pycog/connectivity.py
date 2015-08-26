from __future__ import division

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

THIS = "pycog.connectivity"

class Connectivity(object):
    """
    Constrain the connectivity.

    """
    @staticmethod
    def is_connected(C):
        if nx is None:
            return True

        G = nx.from_numpy_matrix(C, create_using=nx.DiGraph())
        return nx.is_strongly_connected(G)

    def __init__(self, C_or_N, Cfixed=None, p=1, rng=None, seed=4321):
        """
        Initialize.

        Parameters
        ----------
        
        C_or_N : 2D numpy.ndarray or int
                 If ``int``, create a random binary mask with density p.

        Cfixed : 2D numpy.ndarray
                 Fixed weights.

        p : float, optional
            Density of non-self connections.

        rng : numpy.random.RandomState, optional
              Random number generator.

        seed : int, optional
               Seed for random number generator if ``rng`` is not provided.

        """
        if isinstance(C_or_N, int):
            N    = C_or_N
            ntot = N*(N-1)
            nnz  = int(p*ntot)

            if rng is None:
                rng = np.random.RandomState(seed)

            x = np.concatenate((np.ones(nnz, dtype=int), np.zeros(ntot-nnz, dtype=int)))
            rng.shuffle(x)

            C = np.zeros((N, N), dtype=int)
            k = 0
            for i in xrange(N):
                for j in xrange(N):
                    if i == j: continue

                    C[i,j] = x[k]
                    k += 1
        else:
            C = C_or_N

        # Check that a square connection matrix is connected.
        if C.shape[0] == C.shape[1] and not Connectivity.is_connected(C):
            print("[ {}.Connectivity ] Warning: The connection matrix is not connected."
                  .format(THIS))

        self.define(C, Cfixed)

    #/////////////////////////////////////////////////////////////////////////////////////

    def define(self, C, Cfixed):
        """
        C : 2D numpy.ndarray
            Plastic weights

        Cfixed : 2D numpy.ndarray
                 Fixed weights

        """
        # Connectivity properties
        self.C     = C
        self.shape = C.shape
        self.size  = C.size

        # Plastic weights
        self.plastic      = C[np.where(C != 0)]
        self.nplastic     = len(self.plastic)
        self.idx_plastic, = np.where(C.ravel() != 0)

        # Fixed weights
        if Cfixed is not None:
            self.fixed      = Cfixed[np.where(Cfixed != 0)]
            self.nfixed     = len(self.fixed)
            self.idx_fixed, = np.where(Cfixed.ravel() != 0)
        else:
            self.fixed     = np.zeros(0)
            self.nfixed    = 0
            self.idx_fixed = np.zeros(0, dtype=int)

        # Check that indices do not overlap
        assert len(np.intersect1d(self.idx_plastic, self.idx_fixed)) == 0

        self.p_plastic = self.nplastic/self.size
        self.p         = (self.nplastic + self.nfixed)/self.size

        # Mask for plastic weights
        w = np.zeros(C.size)
        w[self.idx_plastic] = 1
        self.mask_plastic = w.reshape(C.shape)

        # Mask for fixed weights
        w = np.zeros(C.size)
        w[self.idx_fixed] = self.fixed
        self.mask_fixed = w.reshape(C.shape)

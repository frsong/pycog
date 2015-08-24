from __future__ import division

import numpy as np

def partition(X, Y, nbins=None, Xedges=None):
    assert nbins is not None or Xedges is not None
    assert len(X) == len(Y)

    if Xedges is None:
        idx     = np.argsort(X)
        Xsorted = X[idx]
        Ysorted = Y[idx]
    
        p = range(0, len(X), len(X)//nbins)
        if len(p) == nbins:
            p.append(len(X))
        else:
            p[-1] = len(X)
        assert len(p) == nbins+1

        Xbins = [Xsorted[p[i]:p[i+1]] for i in xrange(nbins)]
        Ybins = [Ysorted[p[i]:p[i+1]] for i in xrange(nbins)]

        Xedges = np.array([Xsorted[0]]
                          + [(Xbins[i][-1]+Xbins[i+1][0])/2 for i in xrange(nbins-1)]
                          + [Xsorted[-1]])
    else:
        nbins = len(Xedges)-1
        wbins = [np.where((Xedges[i] <= X) & (X < Xedges[i+1]))[0]
                 for i in xrange(nbins-1)]
        wbins.append(np.where((Xedges[nbins-1] <= X) & (X <= Xedges[nbins]))[0])

        Xbins = [X[w] for w in wbins]
        Ybins = [Y[w] for w in wbins]

    binsizes = np.array([len(Xbin) for Xbin in Xbins])
    assert(np.sum(binsizes) == len(X))
    
    return Xbins, Ybins, Xedges, binsizes

from __future__ import division

import numpy as np

from cnslib.figtools import apply_alpha, colorConverter, Figure

#=========================================================================================
# Colors

graylevel = 0.1

#=========================================================================================


#=========================================================================================

def format_cm(plot, fs, lw, ticksize, m, n):
    #plot.equal()

    plot.set_frame_thickness(0.2)
    plot.set_tick_params(ticksize, fs, 0.5)

    plot.ax.xaxis.tick_top()
    plot.ax.xaxis.set_label_position('top')
    plot.ax.yaxis.tick_left()
    plot.ax.yaxis.set_label_position('left')

    #-------------------------------------------------------------------------------------
    # Grid

    xmin = -1
    xmax = n

    ymin = -1
    ymax = m

    #for i in xrange(n):
    #    plot.plot([xmin, xmax], i*np.ones(2), color='0.8', lw=lw, zorder=0)
    #    plot.plot(i*np.ones(2), [xmin, xmax], color='0.8', lw=lw, zorder=0)

    #-------------------------------------------------------------------------------------
    # Area labels

    plot.xticks([])
    #plot.xticklabels(areas, rotation='vertical')

    plot.yticks([])
    #plot.yticklabels(areas)

    #-------------------------------------------------------------------------------------

    #plot.xlim(xmin, xmax)
    #plot.ylim(ymax, ymin)

def plot_cm(plot, C, props, scalarmap_exc, scalarmap_inh):
    ms = props['ms']
    fs = props['fs']
    lw = props['lw']

    ticksize = props['ticksize']

    #-------------------------------------------------------------------------------------

    m, n = C.shape

    Cexc = C[np.where(C > 0)]
    print(np.min(Cexc), np.max(Cexc))

    w = np.where(C < 0)
    if len(w[0]) > 0:
        Cinh = -C[np.where(C < 0)]
        print(np.min(Cinh), np.max(Cinh))
    
    Cdisplay = np.zeros((m, n, 3))
    for i in xrange(m):
        for j in xrange(n):
            if C[i,j] > 0:
                Cdisplay[i,j] = scalarmap_exc.to_rgba(C[i,j])[:3]
            elif C[i,j] < 0:
                Cdisplay[i,j] = scalarmap_inh.to_rgba(-C[i,j])[:3]
            else:
                Cdisplay[i,j] = np.ones(3)

    print(m, n)
    plot.imshow(Cdisplay, interpolation='nearest', aspect='auto')

    #-------------------------------------------------------------------------------------

    format_cm(plot, fs, lw, ticksize, m, n)

#=========================================================================================

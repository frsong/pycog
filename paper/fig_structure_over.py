#! /usr/bin/env python
from __future__ import division

import imp
import os
import sys

import numpy as np

from pycog             import RNN
from pycog.figtools    import gradient, mpl, Figure
from examples.analysis import rdm

import paper

#=========================================================================================
# Paths
#=========================================================================================

here     = os.path.dirname(os.path.realpath(__file__))
figspath = here + '/figs'

nodale_trialsfile = paper.scratchpath + '/rdm_nodale/trials/rdm_nodale_trials.pkl'
dense_trialsfile  = paper.scratchpath + '/rdm_dense_over/trials/rdm_dense_over_trials.pkl'
fixed_trialsfile  = paper.scratchpath + '/rdm_fixed/trials/rdm_fixed_trials.pkl'

# Load model
modelfile   = here + '/../examples/models/rdm_fixed.py'
m_rdm_fixed = imp.load_source('model', modelfile)

#=========================================================================================
# Figure setup
#=========================================================================================

w   = 7.5
h   = 6.5
r   = w/h
fig = Figure(w=w, h=h, axislabelsize=7, labelpadx=6, labelpady=6,
             thickness=0.6, ticksize=3, ticklabelsize=6, ticklabelpad=2)

w_rec = 0.25
h_rec = r*w_rec

w_in  = 0.03
h_in  = h_rec

w_out = w_rec
h_out = r*w_in

hspace = 0.015
vspace = r*hspace

dx = 0.03
x0 = 0.035
x1 = x0 + w_rec + hspace + w_in + dx
x2 = x1 + w_rec + hspace + w_in + dx

offset = 0.03
x0_psy = x0 + offset
x1_psy = x1 + offset
x2_psy = x2 + offset

w_psy = 0.2
h_psy = 0.17

dy = 0.09
y0 = 0.44
y1 = y0 - h_rec - dy

w_mask = 0.12
h_mask = r*w_mask

plots = {
    'Bcon':  fig.add([0.55,  0.77, w_mask, h_mask], None),
    'Bmask': fig.add([0.705, 0.77, w_mask, h_mask], None),
    'Belem': fig.add([0.86,  0.77, w_mask, h_mask], None),

    'Cpsy': fig.add([x0_psy,          y0,              w_psy, h_psy]),
    'Crec': fig.add([x0,              y1,              w_rec, h_rec], None),
    'Cin':  fig.add([x0+w_rec+hspace, y1,              w_in,  h_in],  None), 
    'Cout': fig.add([x0,              y1-vspace-h_out, w_out, h_out], None, 
                    ticklabelpadx=1.5),

    'Dpsy': fig.add([x1_psy,          y0,              w_psy, h_psy]),
    'Drec': fig.add([x1,              y1,              w_rec, h_rec], None),
    'Din':  fig.add([x1+w_rec+hspace, y1,              w_in,  h_in],  None), 
    'Dout': fig.add([x1,              y1-vspace-h_out, w_out, h_out], None),

    'Epsy': fig.add([x2_psy,          y0,              w_psy, h_psy]),
    'Erec': fig.add([x2,              y1,              w_rec, h_rec], None),
    'Ein':  fig.add([x2+w_rec+hspace, y1,              w_in,  h_in],  None), 
    'Eout': fig.add([x2,              y1-vspace-h_out, w_out, h_out], None)
    }

dx2 = x2 - x1
x0  = 0.01
x1  = x0 + dx2
x2  = x1 + dx2
x_mask = 0.525

y0 = 0.97
y1 = 0.64

plotlabels = {
    'A': (x0,     y0),
    'B': (x_mask, y0),
    'C': (x0,     y1),
    'D': (x1,     y1),
    'E': (x2,     y1)
    }
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Load RNNs to compare
#=========================================================================================

savefile      = here + '/../examples/work/data/rdm_nodale/rdm_nodale.pkl'
rnn_nodale    = RNN(savefile, verbose=True)
dprime_nodale = here + '/../examples/work/data/rdm_nodale/rdm_nodale_dprime.txt'
sortby_nodale = here + '/../examples/work/data/rdm_nodale/rdm_nodale_selectivity.txt'

savefile     = here + '/../examples/work/data/rdm_dense_over/rdm_dense_over.pkl'
rnn_dense    = RNN(savefile, verbose=True)
dprime_dense = here + '/../examples/work/data/rdm_dense_over/rdm_dense_over_dprime.txt'
sortby_dense = here + '/../examples/work/data/rdm_dense_over/rdm_dense_over_selectivity.txt'

savefile     = here + '/../examples/work/data/rdm_fixed/rdm_fixed.pkl'
rnn_fixed    = RNN(savefile, verbose=True)
dprime_fixed = here + '/../examples/work/data/rdm_fixed/rdm_fixed_dprime.txt'
sortby_fixed = here + '/../examples/work/data/rdm_fixed/rdm_fixed_selectivity.txt'

#=========================================================================================
# Create color maps for weights
#=========================================================================================

# Colors
white = 'w'
blue  = '#2171b5'
red   = '#cb181d'

def generate_cmap(Ws):
    exc = []
    inh = []
    for W in Ws:
        exc.append( np.ravel(W[np.where(W > 0)]))
        inh.append(-np.ravel(W[np.where(W < 0)]))

    exc = np.sort(np.concatenate(exc))
    inh = np.sort(np.concatenate(inh))

    exc_ignore = int(0.1*len(exc))
    inh_ignore = int(0.1*len(inh))

    cmap_exc = gradient(white, blue)
    norm_exc = mpl.colors.Normalize(vmin=0, vmax=np.round(exc[-exc_ignore], 1), clip=True)
    smap_exc = mpl.cm.ScalarMappable(norm_exc, cmap_exc)

    cmap_inh = gradient(white, red)
    norm_inh = mpl.colors.Normalize(vmin=0, vmax=np.round(inh[-inh_ignore], 1), clip=True)
    smap_inh = mpl.cm.ScalarMappable(norm_inh, cmap_inh)

    cmap_inh_r = gradient(red, white)
    norm_inh_r = mpl.colors.Normalize(vmin=-np.round(inh[-inh_ignore], 1), vmax=0, clip=True)
    smap_inh_r = mpl.cm.ScalarMappable(norm_inh_r, cmap_inh_r)

    return smap_exc, smap_inh

# Color maps
rnns = [rnn_nodale, rnn_dense, rnn_fixed]
smap_exc_in,  smap_inh_in  = generate_cmap([rnn.Win  for rnn in rnns])
smap_exc_rec, smap_inh_rec = generate_cmap([rnn.Wrec for rnn in rnns])
smap_exc_out, smap_inh_out = generate_cmap([rnn.Wout for rnn in rnns])

#=========================================================================================
# Labels
#=========================================================================================

plot = plots['Cpsy']
plot.xlabel(r'Percent coherence toward choice 1')
plot.ylabel(r'Percent choice 1', labelpad=4)
plot.text_upper_center('No Dale\'s principle', dy=0.2, fontsize=7)

plot = plots['Dpsy']
plot.text_upper_center('Dale\'s principle, dense connectivity', dy=0.2, fontsize=7)

plot = plots['Epsy']
plot.text_upper_center('Dale\'s principle, fixed connectivity', dy=0.2, fontsize=7)

#=========================================================================================
# Psychometric curves
#=========================================================================================

# No Dale
plot = plots['Cpsy']
rdm.psychometric_function(nodale_trialsfile, plot, ms=5)

# Dale, dense
plot = plots['Dpsy']
rdm.psychometric_function(dense_trialsfile, plot, ms=5)

# Dale, fixed
plot = plots['Epsy']
rdm.psychometric_function(fixed_trialsfile, plot, ms=5)

#=========================================================================================
# Masking illustration
#=========================================================================================

plot = plots['Bcon']
RNN.plot_connection_matrix(plot, rnn_fixed.Wrec, smap_exc_rec, smap_inh_rec)

plot = plots['Bmask']
W = m_rdm_fixed.Crec*m_rdm_fixed.ei
RNN.plot_connection_matrix(plot, np.sign(W), smap_exc_rec, smap_inh_rec)

plot = plots['Belem']
smap_exc = mpl.cm.ScalarMappable(smap_exc_rec.norm, mpl.cm.gray_r)
smap_inh = mpl.cm.ScalarMappable(smap_inh_rec.norm, mpl.cm.gray_r)
RNN.plot_connection_matrix(plot, abs(rnn_fixed.Wrec), smap_exc, smap_inh)

#-----------------------------------------------------------------------------------------
# Labels
#-----------------------------------------------------------------------------------------

plot = plots['Bcon']
plot.text(1.09, 0.5, '=', ha='left', va='center', fontsize=10, 
          transform=plot.transAxes)
plot.text(0.5, +1.05, 'Pre', ha='center', va='bottom', fontsize=7, 
          transform=plot.transAxes)
plot.text(-0.05, 0.5, 'Post', ha='right', va='center', fontsize=7, 
          transform=plot.transAxes, rotation='vertical')
plot.text(0.5, -0.15, r'$W^\text{rec}$', ha='center', va='top', fontsize=10, 
          transform=plot.transAxes)

plot = plots['Bmask']
plot.text(1.09, 0.5, r'$\odot$', ha='left', va='center', fontsize=10, 
          transform=plot.transAxes)
plot.text(0.5, +1.2, 'Fixed structure mask', ha='center', va='top', fontsize=7,
          transform=plot.transAxes)
plot.text(0.5, -0.15, r'$M^\text{rec}$', ha='center', va='top', fontsize=10, 
          transform=plot.transAxes)

plot = plots['Belem']
plot.text(0.5, +1.2, 'Trained positive weights', ha='center', va='top', fontsize=7, 
          transform=plot.transAxes)
plot.text(0.5, -0.15, r'$W^\text{rec,+}$', ha='center', va='top', fontsize=10, 
          transform=plot.transAxes)

#=========================================================================================
# Connection matrices
#=========================================================================================

for rnn, sortbyfile, s, dprimefile in zip([rnn_nodale, rnn_dense, rnn_fixed],
                                          [sortby_nodale, sortby_dense, sortby_fixed],
                                          ['C', 'D', 'E'],
                                          [dprime_nodale, dprime_dense, dprime_fixed]):
    idx = np.loadtxt(sortbyfile, dtype=int)
    RNN.plot_connection_matrix(plots[s+'in'], rnn.Win[idx,:],
                               smap_exc_in, smap_inh_in)
    RNN.plot_connection_matrix(plots[s+'rec'], rnn.Wrec[idx,:][:,idx],
                               smap_exc_rec, smap_inh_rec)
    RNN.plot_connection_matrix(plots[s+'out'], rnn.Wout[:,idx],
                               smap_exc_out, smap_inh_out)

    dprime = np.loadtxt(dprimefile)
    transitions = []
    for i in xrange(1, len(dprime)):
        if dprime[i-1] > 0 and dprime[i] < 0:
            transitions.append(i)

    plot = plots[s+'rec']
    if s == 'C':
        n = transitions[0]
        plot.text(n-0.5, -1.5, '|', ha='center', va='center', fontsize=8)
        plot.text(n-0.5-0.5, -1.3, r'$\leftarrow$\ Selective for 1',
                  ha='right', va='center', fontsize=6)
        plot.text(n-0.5+0.5, -1.3, r'Selective for 2 $\rightarrow$',
                  ha='left', va='center', fontsize=6)
    else:
        # Excitatory units
        n = transitions[0]
        plot.text(n-0.5, -1.5, '|', ha='center', va='center', color=Figure.colors('blue'),
                  fontsize=8)
        plot.text(n-0.5-0.5, -1.5, r'$\leftarrow$', color=Figure.colors('blue'),
                  ha='right', va='center', fontsize=6)
        plot.text(n-0.5+0.5, -1.5, r'$\rightarrow$', color=Figure.colors('blue'),
                  ha='left', va='center', fontsize=6)

        # Inhibitory units
        n = transitions[1]
        plot.text(n-0.5, -1.5, '|', ha='center', va='center', color=Figure.colors('red'),
                  fontsize=8)
        plot.text(n-0.5-0.5, -1.5, r'$\leftarrow$', color=Figure.colors('red'),
                  ha='right', va='center', fontsize=6)
        plot.text(n-0.5+0.5, -1.5, r'$\rightarrow$', color=Figure.colors('red'),
                  ha='left', va='center', fontsize=6)

plot = plots['Cin']
plot.xaxis.set_tick_params(pad=-3)
plot.xticks(np.arange(2)+0.1)
plot.xticklabels(['Choice 1', 'Choice 2'], rotation='vertical', fontsize=5.5)


plot = plots['Cout']
plot.yaxis.set_tick_params(pad=-2)
plot.yticks(np.arange(2)+0.14)
plot.yticklabels(['Choice 1', 'Choice 2'], fontsize=5.5)

#=========================================================================================

fig.save(path=figspath)

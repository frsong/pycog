#! /usr/bin/env python
from __future__ import division

import cPickle as pickle
import imp
import os
from   os.path import join

import numpy as np

from pycog             import RNN
from pycog.figtools    import Figure, mpl, PCA
from pycog.utils       import get_here, get_parent
from examples.analysis import lee

import paper

#=========================================================================================
# Paths
#=========================================================================================

here     = get_here(__file__)
base     = get_parent(here)
figspath = join(here, 'figs')

savefile   = join(base, 'examples', 'work', 'data', 'lee', 'lee.pkl')
trialsfile = join(paper.scratchpath, 'lee', 'trials', 'lee_trials.pkl')

# Model
modelfile = join(base, 'examples', 'models', 'lee.py')
m         = imp.load_source('model', modelfile)

#=========================================================================================
# Figure setup
#=========================================================================================

w   = 7.5
h   = 6.5
r   = w/h
fig = Figure(w=w, h=h, axislabelsize=7, labelpadx=5, labelpady=6,
             thickness=0.6, ticksize=3, ticklabelsize=6, ticklabelpad=2,
             format=paper.format)

w    = 0.095
dx   = w + 0.02
x0   = 0.07
xall = [x0+i*dx for i in xrange(m.nseq)]

h  = 0.1
dy = 0.05
y0 = 0.55
y1 = y0 - h - dy - 0.05
y2 = y1 - h - dy
y3 = y2 - h - dy

y_task = 0.85
w_task = 0.42
h_task = 0.1

w_pca = 0.4
h_pca = 0.38
x_pca = x0 + w_task + 0.09
y_pca = 0.555

y_seq = y_pca
w_seq = w_task
h_seq = 0.12

y_dots = y_seq + h_seq + 0.015
w_dots = w_seq
h_dots = h_seq

x_cbar = x0 + w_seq + 0.015
y_cbar = y_seq
w_cbar = 0.015
h_cbar = h_seq

x_task_screen = x0 + w_task*3/7
y_task_screen = 0.915
w_task_screen = w_task/7
h_task_screen = r*w_task_screen

plots = {}
plots['task'] = fig.add([x0, y_task, w_task, h_task], style='none')
plots['task_fixation'] = fig.add([x_task_screen - w_task_screen/2,
                                  y_task_screen, w_task_screen, h_task_screen], 'none')
plots['task_M1']       = fig.add([x_task_screen + w_task_screen,
                                  y_task_screen, w_task_screen, h_task_screen], 'none')
plots['task_M2']       = fig.add([x_task_screen + 1.8*w_task_screen,
                                  y_task_screen, w_task_screen, h_task_screen], 'none')
plots['task_M3']       = fig.add([x_task_screen + 3*w_task_screen,
                                  y_task_screen, w_task_screen, h_task_screen], 'none')

plots['pca']  = fig.add([x_pca,  y_pca,  w_pca,  h_pca])
plots['dots'] = fig.add([x0,     y_dots, w_dots, h_dots])
plots['seq']  = fig.add([x0,     y_seq,  w_seq,  h_seq])
plots['cbar'] = fig.add([x_cbar, y_cbar, w_cbar, h_cbar])
for i, x in enumerate(xall):
    plots[str(i)+'_screen'] = fig.add([x-0.001, y1-0.01, w, h], style='none')
    plots[str(i)+'_x']      = fig.add([x,       y2,      w, h])
    plots[str(i)+'_y']      = fig.add([x,       y3+0.02, w, h])

x0 = 0.01
x1 = 0.525
y0 = 0.97
y1 = 0.81
y2 = 0.5

plotlabels = {
    'A': (x0, y0),
    'B': (x0, y1),
    'C': (x1, y0),
    'D': (x0, y2),
    }
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

R_target = 0.5
r_target = 0.1
r_screen = R_target + 2*r_target

#=========================================================================================
# Dot colors
#=========================================================================================

colors = {
    'iti':      '0.6',
    'fixation': Figure.colors('orange'),
    'M1':       Figure.colors('magenta'),
    'M2':       Figure.colors('green'),
    'M3':       Figure.colors('purple')
}

colors = {
    0: Figure.colors('blue'),
    1: Figure.colors('red'),
    2: Figure.colors('green'),
    3: Figure.colors('orange'),
    4: Figure.colors('0.2'),
    5: Figure.colors('purple'),
    6: Figure.colors('magenta'),
    7: Figure.colors('darkblue'),
    8: Figure.colors('aquamarine')
}

#=========================================================================================
# Labels
#=========================================================================================

plot = plots['dots']
plot.ylabel('Dot')

plot = plots['seq']
plot.xlabel('Time (sec)')
plot.ylabel('Sequence')

plot = plots['0_x']
plot.ylabel('$x$-position')

plot = plots['0_y']
plot.xlabel('Time (sec)')
plot.ylabel('$y$-position')

#=========================================================================================
# Task structure
#=========================================================================================

plot = plots['task']

lw    = 1
bl    = 0.4
hh    = 0.05
color = '0.2'

# All time
plot.plot([0, 3.5], bl*np.ones(2), color=color, lw=lw)

# Markers
plot.plot(0.0*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)
plot.plot(1.0*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)
plot.plot(2.0*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)
plot.plot(2.5*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)
plot.plot(3.0*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)
plot.plot(3.5*np.ones(2), [bl-hh, bl+hh], color=color, lw=lw)

# Epoch labels
p = {'fontsize': 7, 'ha': 'center', 'va': 'top', 'transform': plot.ax.transData}
ytext = bl - 4*hh
plot.text(0.50, ytext, 'Intertrial interval (ITI)', **p)
plot.text(1.50, ytext, 'Fixation', **p)
plot.text(2.25, ytext, 'Hold',     **p)
plot.text(2.75, ytext, 'Hold',     **p)
plot.text(3.25, ytext, 'Hold',     **p)

# Movement labels
p = {'fontsize': 7, 'ha': 'center', 'va': 'top', 'transform': plot.ax.transData}
ytext = bl - 10*hh
plot.text(2.0, ytext, 'M1', **p)
plot.text(2.5, ytext, 'M2', **p)
plot.text(3.0, ytext, 'M3', **p)

# Movement arrows
p = dict(head_width=0.05, head_length=0.1, fc='k', ec='k')
plot.arrow(2.0, ytext + 2*hh, 0, 3*hh, **p)
plot.arrow(2.5, ytext + 2*hh, 0, 3*hh, **p)
plot.arrow(3.0, ytext + 2*hh, 0, 3*hh, **p)

plot.xlim(0, 3.5)
plot.ylim(0, 1)

# Fixation screen
plot = plots['task_fixation']
plot.equal()
for k in [4]:
    color = Figure.colors('Blue')
    plot.circle(m.target_position(k), 0.8*r_target, ec='none', fc=color, alpha=0.6)
plot.xlim(-r_screen, r_screen)
plot.ylim(-r_screen, r_screen)

# Arrow properties
arrowp = dict(head_width=0.1, head_length=0.1, lw=1,
              fc=Figure.colors('strongblue'), ec=Figure.colors('strongblue'))
alen   = 0.22

# M1 screen
plot = plots['task_M1']
plot.equal()
for k in [3, 4, 5]:
    color = Figure.colors('Blue') if k == 5 else Figure.colors('blue')
    plot.circle(m.target_position(k), 0.8*r_target, ec='none', fc=color, alpha=0.6)
pos = m.target_position(4)
plot.arrow(pos[0], pos[1], alen, 0, **arrowp)
plot.xlim(-r_screen, r_screen)
plot.ylim(-r_screen, r_screen)

# M2 screen
plot = plots['task_M2']
plot.equal()
for k in [1, 5, 7]:
    color = Figure.colors('Blue') if k == 1 else Figure.colors('blue')
    plot.circle(m.target_position(k), 0.8*r_target, ec='none', fc=color, alpha=0.6)
pos = m.target_position(5)
plot.arrow(pos[0], pos[1], -1.2*alen, 1.2*alen, **arrowp)
plot.xlim(-r_screen, r_screen)
plot.ylim(-r_screen, r_screen)

# M3 screen
plot = plots['task_M3']
plot.equal()
for k in [0, 1, 2]:
    color = Figure.colors('Blue') if k == 2 else Figure.colors('blue')
    plot.circle(m.target_position(k), 0.8*r_target, ec='none', fc=color, alpha=0.6)
pos = m.target_position(1)
plot.arrow(pos[0], pos[1], alen, 0, **arrowp)
plot.xlim(-r_screen, r_screen)
plot.ylim(-r_screen, r_screen)

#=========================================================================================
# PCA
#=========================================================================================

plot = plots['pca']

#-----------------------------------------------------------------------------------------
# PCA analysis
#-----------------------------------------------------------------------------------------

# RNN
rnn = RNN(savefile, {'dt': 0.5}, verbose=False)

# Run each sequence separately
rnn.p['mode'] = None

# Turn off noise
rnn.p['var_in']  = 0
rnn.p['var_rec'] = 0

dt_save = 2
trials  = {}
for seq in xrange(1, 1+m.nseq):
    print('Sequence #{}, noiseless'.format(seq))

    # Trial
    trial_func = m.generate_trial
    trial_args = {'name': 'test', 'seq':  seq}
    info = rnn.run(inputs=(trial_func, trial_args), seed=10)

    # Save at lower temporal resolution
    dt    = rnn.t[1] - rnn.t[0]
    step  = int(dt_save/dt)
    trial = {
        't':      rnn.t[::step],
        'r':      rnn.r[:,::step],
        'epochs': info['epochs']
        }
    trials[seq] = trial

# PCA
pca, active_units = lee.pca_analysis(trials)

#-----------------------------------------------------------------------------------------
# PCA dimensionality reduction
#-----------------------------------------------------------------------------------------

xall   = []
yall   = []
labels = []
for seq in xrange(1, 1+m.nseq):
    trial = trials[seq]
    e     = trial['epochs']
    t     = trial['t']
    X     = trial['r'].T[:,active_units]
    Xpca  = pca.project(X)

    # Epochs
    fix, = np.where((e['fixation'][0] < t) & (t <= e['fixation'][1]))
    M1,  = np.where((e['M1'][0]       < t) & (t <= e['M1'][1]))
    M2,  = np.where((e['M2'][0]       < t) & (t <= e['M2'][1]))
    M3,  = np.where((e['M3'][0]       < t) & (t <= e['M3'][1]))

    fix_start = fix[0]
    M1_start  = M1[0]
    M2_start  = M2[0]
    M3_start  = M3[0]

    # Trajectory
    x = Xpca[:,0]; xall.append(x)
    y = Xpca[:,1]; yall.append(y)

    # Dots
    dot0, dot1, dot2, dot3 = m.sequences[seq]

    # Starting point
    plot.plot(x[M1[0]], y[M1[0]], 'o', mfc='none', mec=colors[dot0], ms=4, mew=1,
              zorder=20)

    # Ending point
    plot.plot(x[M3[-1]], y[M3[-1]], 'o', mfc=colors[dot3], mec=colors[dot3], ms=3, mew=1)

    # Epochs
    plot.plot(x[M1], y[M1], color=colors[dot1], lw=0.75)
    plot.plot(x[M2], y[M2], color=colors[dot2], lw=0.75)
    plot.plot(x[M3], y[M3], color=colors[dot3], lw=0.75)

    # Offsets
    offsets = {
        1: (+0.7, -0.5),
        5: (+0.9, +0.5),
        7: (-0.7, +0.5),
        8: (-0.8, +0.5)
        }

    # Endpoint
    delta_x, delta_y = offsets.get(seq, (-1.2, 0))
    plot.text(x[M3[-1]]+delta_x, y[M3[-1]]+delta_y, r'\#{{{}}}'.format(seq), zorder=15,
              color='0.2', ha='center', va='center', fontsize=7,
              transform=plot.ax.transData)

for dot, color in colors.items():
    if dot == 4:
        plot.plot(100, 100, 'o', mfc='none', mec=colors[dot], ms=4, mew=1,
                  label='Dot {}'.format(dot+1))
    else:
        plot.plot(100, 100, 'o', mfc=colors[dot], mec=colors[dot], ms=3, mew=1,
                  label='Dot {}'.format(dot+1))

plot.lim('x', xall)
plot.lim('y', yall)

plot.xlabel('PC 1')
plot.ylabel('PC 2')

# Legend
props = {'prop': {'size': 5.5}, 'handlelength': 1,
         'handletextpad': 1, 'labelspacing': 0.5}
plot.legend(bbox_to_anchor=(0.7, 1), **props)

# Variance explained
pct_var = 100*np.sum(pca.fracs[:2])
print("First two PCs explains {:.2f}% of the variance.".format(pct_var))

#=========================================================================================
# x-coordinate, y-coordinate, screen
#=========================================================================================

# Load trials
with open(trialsfile) as f:
    trials = pickle.load(f)
#trials = trials[m.nseq:]

for i in xrange(m.nseq):
    trial  = trials[i]
    info   = trial['info']
    epochs = info['epochs']
    seq    = info['seq']
    t      = trial['t']; tsecs = 1e-3*t
    z      = trial['z']

    print("Sequence #{}".format(seq))

    # Separate the ITI
    iti,     = np.where(t <= epochs['iti'][1])
    not_iti, = np.where(t >  epochs['iti'][1])

    # Colors
    color_iti     = '0.6'
    color_not_iti = Figure.colors('strongblue')

    #-------------------------------------------------------------------------------------
    # Trial epochs
    #-------------------------------------------------------------------------------------

    plot = plots[str(i)+'_x']

    annot = {
        'iti':      ('ITI', '0.6'),
        'fixation': ('F',   '#66c2a5'),
        'M1':       ('M1',  '#fc8d62'),
        'M2':       ('M2',  '#8da0cb'),
        'M3':       ('M3',  '#e78ac3')
        }
    idx = {e: np.where((epochs[e][0] < t) & (t <= epochs[e][1]))[0] for e in annot}

    for e in ['iti', 'fixation', 'M1', 'M2', 'M3']:
        yl = 1.1
        yt = 1.15

        if e == 'iti':
            lw = 1
        else:
            lw = 2.5

        label, color = annot[e]

        # Duration
        epoch = np.array(epochs[e])/t[-1]
        plot.plot(epoch, yl*np.ones(2), transform=plot.transAxes, lw=lw, color=color)

        # Label
        if i == 0:
            plot.text(np.mean(epoch), yt, label, ha='center', va='bottom',
                      fontsize=4, color=color, transform=plot.transAxes)

    #-------------------------------------------------------------------------------------
    # Sequence #
    #-------------------------------------------------------------------------------------

    plot = plots[str(i)+'_screen']
    plot.text_upper_center('Sequence \#{}'.format(seq), dy=0.2, fontsize=6.5)

    #-------------------------------------------------------------------------------------
    # x-coordinate
    #-------------------------------------------------------------------------------------

    plot = plots[str(i)+'_x']

    if i > 0:
        plot.plot(tsecs[idx['iti']], z[0,idx['iti']], color=annot['iti'][1], lw=1)
    for e in annot:
        if e == 'iti': continue
        plot.plot(tsecs[idx[e]], z[0,idx[e]], color=annot[e][1], lw=1)

    plot.xlim(tsecs[0], tsecs[-1])
    plot.xticks(np.arange(tsecs[0], tsecs[-1], 1))
    plot.xticklabels()
    plot.ylim(-r_screen, r_screen)
    plot.yticks([-R_target, 0, R_target])

    #-------------------------------------------------------------------------------------
    # y-coordinate
    #-------------------------------------------------------------------------------------

    plot = plots[str(i)+'_y']

    if i > 0:
        plot.plot(tsecs[idx['iti']], z[1,idx['iti']], lw=1, color=annot['iti'][1])
    for e in annot:
        if e == 'iti': continue
        plot.plot(tsecs[idx[e]], z[1,idx[e]], color=annot[e][1], lw=1)

    plot.xlim(tsecs[0], tsecs[-1])
    plot.xticks(np.arange(tsecs[0], tsecs[-1], 1))
    plot.ylim(-r_screen, r_screen)
    plot.yticks([-R_target, 0, R_target])

    #-------------------------------------------------------------------------------------
    # Screen
    #-------------------------------------------------------------------------------------

    plot = plots[str(i)+'_screen']
    plot.equal()

    # Target dots
    for k in m.DOTS:
        plot.circle(m.target_position(k), r_target,
                    ec='none', fc=Figure.colors('blue'), alpha=0.6)

    # Eye position on screen
    if i > 0:
        plot.plot(z[0,iti], z[1,iti],     color=color_iti,     lw=1.25)
    plot.plot(z[0,not_iti], z[1,not_iti], color=color_not_iti, lw=1.25)

    # Dot labels
    if i == 0:
        for k in m.DOTS:
            p = m.target_position(k)
            plot.text(p[0], p[1]+0.14, str(k+1), ha='center', va='bottom',
                      color='k', fontsize=5)

    plot.xlim(-r_screen, r_screen)
    plot.ylim(-r_screen, r_screen)

#-----------------------------------------------------------------------------------------
# Shared axes
#-----------------------------------------------------------------------------------------

for i in xrange(1, m.nseq):
    plots[str(i)+'_x'].axis_off('left')
    plots[str(i)+'_y'].axis_off('left')
    plots[str(i)+'_screen'].xticks()
    plots[str(i)+'_screen'].yticks()

#=========================================================================================
# Inputs
#=========================================================================================

for trial in trials:
    if trial['info']['seq'] == 5:
        break
t = 1e-3*trial['t']
u = trial['u']

#-----------------------------------------------------------------------------------------
# Color map
#-----------------------------------------------------------------------------------------

vmax = np.max(u[m.DOTS + m.SEQUENCE])
vmax = np.round(vmax, 1)
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=0, vmax=vmax, clip=True)

#-----------------------------------------------------------------------------------------
# Color bar
#-----------------------------------------------------------------------------------------

plot = plots['cbar']
cbar = mpl.colorbar.ColorbarBase(plot.ax, cmap=cmap, norm=norm, orientation='vertical')

w = 0.25
cbar.outline.set_linewidth(w)
plot.yaxis.set_tick_params(width=w, size=2, labelsize=6, pad=-2)

cbar.set_ticks([0, vmax])
cbar.solids.set_edgecolor("face") # Correct stripes

#-----------------------------------------------------------------------------------------
# Inputs
#-----------------------------------------------------------------------------------------

for p, idx in zip(['dots', 'seq'], [m.DOTS, m.SEQUENCE]):
    plot = plots[p]
    plot.imshow(u[idx], aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                origin='lower', extent=[0, t[-1], -0.5, len(idx)-0.5])
    plot.yticks(range(len(idx)))
    plot.yticklabels(range(1, 1+len(idx)))
plots['dots'].xticks()

#=========================================================================================

fig.save(path=figspath)

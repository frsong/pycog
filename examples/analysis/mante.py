from __future__ import division

import cPickle as pickle
import os
import sys
from   os.path import join

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as smooth

from pycog          import fittools, RNN, tasktools
from pycog.figtools import apply_alpha, Figure

THIS = "examples.analysis.mante"

#=========================================================================================
# Setup
#=========================================================================================

# File to store trials in
def get_trialsfile(p):
    return join(p['trialspath'], p['name'] + '_trials.pkl')

# Load trials
def load_trials(trialsfile):
    with open(trialsfile) as f:
        trials = pickle.load(f)

    return trials, len(trials)

# File to store sorted trials in
def get_sortedfile(p):
    return join(p['trialspath'], p['name'] + '_sorted.pkl')

# File to store regression coefficients in
def get_betafile(p):
    return join(p['datapath'], p['name'] + '_beta.pkl')

# Simple choice function
def get_choice(trial):
    return np.argmax(trial['z'][:,-1])

# Define "active" units
def is_active(r):
    return np.std(r) > 0.1

# Coherence scale
SCALE = 5

#=========================================================================================
# Trials
#=========================================================================================

def run_trials(p, args):
    # Model
    m = p['model']

    #cohs = [1, 4, 16]
    #nconditions = len(m.contexts)*(len(cohs)*len(m.left_rights))**2

    # In case we want to customize
    cohs = m.cohs
    nconditions = m.nconditions

    # Number of trials
    try:
        ntrials = int(args[0])
    except:
        ntrials = 100
    ntrials *= nconditions

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            # Condition
            k = tasktools.unravel_index(i % nconditions,
                                        (len(cohs), len(m.left_rights),
                                         len(cohs), len(m.left_rights),
                                         len(m.contexts)))
            coh_m        = cohs[k[0]]
            left_right_m = m.left_rights[k[1]]
            coh_c        = cohs[k[2]]
            left_right_c = m.left_rights[k[3]]
            context      = m.contexts[k[4]]

            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':         'test',
                'catch':        False,
                'coh_m':        coh_m,
                'left_right_m': left_right_m,
                'coh_c':        coh_c,
                'left_right_c': left_right_c,
                'context':      context
                }
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            s = ("Trial {:>{}}/{}: ({}) m{:>+3}, c{:>+3}"
                 .format(i+1, w, ntrials, info['context'],
                         info['left_right_m']*info['coh_m'],
                         info['left_right_c']*info['coh_c']))
            sys.stdout.write(backspaces*'\b' + s)
            sys.stdout.flush()
            backspaces = len(s)

            # Save
            dt    = rnn.t[1] - rnn.t[0]
            step  = int(p['dt_save']/dt)
            trial = {
                't':    rnn.t[::step],
                'u':    rnn.u[:,::step],
                'r':    rnn.r[:,::step],
                'z':    rnn.z[:,::step],
                'info': info,
                }
            trials.append(trial)
    except KeyboardInterrupt:
        pass
    print("")

    # Save all
    filename = get_trialsfile(p)
    with open(filename, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(filename)*1e-9
    print("[ {}.run_trials ] Trials saved to {} ({:.1f} GB)".format(THIS, filename, size))

    # Compute the psychometric function
    psychometric_function(filename)

#=========================================================================================

def psychometric_function(trialsfile, plots=None, **kwargs):
    """
    Psychometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric function
    #-------------------------------------------------------------------------------------

    results  = {cond: {} for cond in ['mm', 'mc', 'cm', 'cc']}
    ncorrect = 0
    for trial in trials:
        info   = trial['info']
        coh_m  = info['left_right_m']*info['coh_m']
        coh_c  = info['left_right_c']*info['coh_c']
        choice = get_choice(trial)

        if choice == info['choice']:
            ncorrect += 1

        if info['context'] == 'm':
            results['mm'].setdefault(coh_m, []).append(choice)
            results['mc'].setdefault(coh_c, []).append(choice)
        else:
            results['cm'].setdefault(coh_m, []).append(choice)
            results['cc'].setdefault(coh_c, []).append(choice)
    print("[ {}.psychometric_function ] {:.2f}% correct."
          .format(THIS, 100*ncorrect/ntrials))

    for cond in results:
        choice_by_coh = results[cond]

        cohs = np.sort(np.array(choice_by_coh.keys()))
        p0   = np.zeros(len(cohs))
        for i, coh in enumerate(cohs):
            choices = np.array(choice_by_coh[coh])
            p0[i]   = 1 - np.sum(choices)/len(choices)
        scaled_cohs = SCALE*cohs

        results[cond] = (scaled_cohs, p0)

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plots is not None:
        ms      = kwargs.get('ms', 5)
        color_m = '0.2'
        color_c = Figure.colors('darkblue')

        for cond, result in results.items():
            # Context
            if cond[0] == 'm':
                color = color_m
                label = 'Motion context'
            else:
                color = color_c
                label = 'Color context'

            # Stimulus
            if cond[1] == 'm':
                plot = plots['m']
            else:
                plot = plots['c']

            # Result
            scaled_cohs, p0 = result

            # Data points
            plot.plot(scaled_cohs, 100*p0, 'o', ms=ms, mew=0, mfc=color, zorder=10)

            # Fit
            try:
                popt, func = fittools.fit_psychometric(scaled_cohs, p0)

                fit_cohs = np.linspace(min(scaled_cohs), max(scaled_cohs), 201)
                fit_p0   = func(fit_cohs, **popt)
                plot.plot(fit_cohs, 100*fit_p0, color=color, lw=1, zorder=5, label=label)
            except RuntimeError:
                print("[ {}.psychometric_function ]".format(THIS)
                      + " Unable to fit, drawing a line through the points.")
                plot.plot(scaled_cohs, 100*p0, color=color, lw=1, zorder=5, label=label)

            plot.lim('x', scaled_cohs)
            plot.ylim(0, 100)

    #-------------------------------------------------------------------------------------

    return results

#=========================================================================================

def get_choice_selectivity(trials):
    """
    Compute d' for choice.

    """
    N     = trials[0]['r'].shape[0]
    Xin   = np.zeros(N)
    Xin2  = np.zeros(N)
    Xout  = np.zeros(N)
    Xout2 = np.zeros(N)
    n_in  = 0
    n_out = 0
    for trial in trials:
        t = trial['t']
        start, end = trial['info']['epochs']['stimulus']
        stimulus,  = np.where((start < t) & (t <= end))

        r = np.sum(trial['r'][:,stimulus], axis=1)

        choice = get_choice(trial)
        if choice == 0:
            Xin  += r
            Xin2 += r**2
            n_in += 1
        else:
            Xout  += r
            Xout2 += r**2
            n_out += 1
    mean_in  = Xin/n_in
    var_in   = Xin2/n_in - mean_in**2
    mean_out = Xout/n_out
    var_out  = Xout2/n_out - mean_out**2
    dprime   = (mean_in - mean_out)/np.sqrt((var_in + var_out)/2)

    return dprime

def get_preferred_targets(trials):
    """
    Determine preferred targets.

    """
    dprime = get_choice_selectivity(trials)

    return 2*(dprime > 0) - 1

#=========================================================================================
# Sort
#=========================================================================================

def sort_func(s, preferred_targets, target, trial):
    choices = preferred_targets*target
    info    = trial['info']
    correct = +1 if get_choice(trial) == info['choice'] else -1

    if s == 'choice':
        return [(choice,) for choice in choices]
    elif s == 'motion_choice':
        cohs = preferred_targets*info['left_right_m']*info['coh_m']
        return [(choice, coh, info['context']) for choice, coh in zip(choices, cohs)]
    elif s == 'colour_choice':
        cohs = preferred_targets*info['left_right_c']*info['coh_c']
        return [(choice, coh, info['context']) for choice, coh in zip(choices, cohs)]
    elif s == 'context_choice':
        return [(choice, info['context']) for choice in choices]
    elif s == 'all':
        cohs_m = preferred_targets*info['left_right_m']*info['coh_m']
        cohs_c = preferred_targets*info['left_right_c']*info['coh_c']
        return [(choice, coh_m, coh_c, info['context'], correct)
                for choice, coh_m, coh_c in zip(choices, cohs_m, cohs_c)]
    else:
        raise ValueError("[ {}.sort_func ] Unknown criterion for sorting.".format(THIS))

def _safe_divide(x):
    if x == 0:
        return 0
    return 1/x

def safe_divide(X):
    return np.array([_safe_divide(x) for x in X])

def sort_trials(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Preferred targets
    preferred_targets = get_preferred_targets(trials)

    # Smoothing parameter
    t  = trials[0]['t']
    dt = t[1] - t[0]
    sigma_smooth = int(50/dt)

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    sortby = ['all', 'choice', 'motion_choice', 'colour_choice', 'context_choice']

    sorted_trials = {s: {} for s in sortby}
    ncorrect = 0
    for i, trial in enumerate(trials):
        choice = get_choice(trial)
        if choice == 0:
            target = +1
        else:
            target = -1

        for s in ['all']:
            sorted_trial = sort_func(s, preferred_targets, target, trial)
            for unit, cond in enumerate(sorted_trial):
                sorted_trials[s].setdefault(cond, []).append((i, unit))

        if choice == trial['info']['choice']:
            ncorrect += 1
            for s in sortby:
                if s in ['all']:
                    continue
                sorted_trial = sort_func(s, preferred_targets, target, trial)
                for unit, cond in enumerate(sorted_trial):
                    sorted_trials[s].setdefault(cond, []).append((i, unit))
    print("[ {}.sort_trials ] {:.2f}% correct.".format(THIS, 100*ncorrect/ntrials))

    #-------------------------------------------------------------------------------------
    # Average within conditions
    #-------------------------------------------------------------------------------------

    nunits, ntime = trial['r'].shape
    for s in sorted_trials:
        # Average
        for cond, i_unit in sorted_trials[s].items():
            r = np.zeros((nunits, ntime))
            n = np.zeros(nunits)
            for i, unit in i_unit:
                r[unit] += trials[i]['r'][unit]
                n[unit] += 1
            r = r*np.tile(safe_divide(n), (ntime, 1)).T
            sorted_trials[s][cond] = smooth(r, sigma_smooth, axis=1)

        # Normalize
        X  = 0
        X2 = 0
        n  = 0
        for cond, r in sorted_trials[s].items():
            X  += np.sum(r,    axis=1)
            X2 += np.sum(r**2, axis=1)
            n  += r.shape[1]
        mean = X/n
        std  = np.sqrt(X2/n - mean**2)

        mean = np.tile(mean, (ntime, 1)).T
        std  = np.tile(std,  (ntime, 1)).T
        for cond, r in sorted_trials[s].items():
            sorted_trials[s][cond] = (r - mean)/std

    #-------------------------------------------------------------------------------------
    # Save
    #-------------------------------------------------------------------------------------

    with open(sortedfile, 'wb') as f:
        pickle.dump((t, sorted_trials), f, pickle.HIGHEST_PROTOCOL)
    print("[ {}.sort_trials ] Sorted trials saved to {}".format(THIS, sortedfile))

#=========================================================================================
# Single-unit activity
#=========================================================================================

def get_active_units(trialsfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)
    trial = trials[0]

    N = trial['r'].shape[0]
    r = np.zeros_like(trial['r'])
    for trial in trials:
        r += trial['r']
    r /= ntrials

    return sorted([i for i in xrange(N) if is_active(r[i])])

def plot_unit(unit, sortedfile, plots, t0=0, tmin=-np.inf, tmax=np.inf, **kwargs):
    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    #-------------------------------------------------------------------------------------
    # Labels
    #-------------------------------------------------------------------------------------

    # Unit no.
    fontsize = kwargs.get('unit_fontsize', 7)
    plots['choice'].text_upper_center('Unit '+str(unit), dy=0.07, fontsize=fontsize)

    # Sort-by
    if kwargs.get('sortby_fontsize') is not None:
        fontsize = kwargs['sortby_fontsize']
        labels = {
            'choice':         'choice',
            'motion_choice':  'motion \& choice',
            'colour_choice':  'color \& choice',
            'context_choice': 'context \& choice'
            }
        for k , label in labels.items():
            plots[k].ylabel(label)

    #-------------------------------------------------------------------------------------
    # Setup
    #-------------------------------------------------------------------------------------

    # Duration to plot
    w, = np.where((tmin <= t) & (t <= tmax))
    t  = t - t0

    # Linestyle
    def get_linestyle(choice):
        if choice == +1:
            return '-'
        return '--'

    # Line width
    lw = kwargs.get('lw', 1)

    # For setting axis limits
    yall = []

    #-------------------------------------------------------------------------------------
    # Choice
    #-------------------------------------------------------------------------------------

    plot = plots['choice']
    condition_averaged = sorted_trials['choice']

    for (choice,), r in condition_averaged.items():
        ls = get_linestyle(choice)
        plot.plot(t[w], r[unit,w], ls, color=Figure.colors('red'), lw=lw)
        yall.append(r[unit,w])
    plot.xlim(t[w][0], t[w][-1])
    plot.xticks([t[w][0], 0, t[w][-1]])

    #-------------------------------------------------------------------------------------
    # Motion & choice
    #-------------------------------------------------------------------------------------

    plot = plots['motion_choice']
    condition_averaged = sorted_trials['motion_choice']

    abscohs = []
    for (choice, coh, context) in condition_averaged:
        abscohs.append(abs(coh))
    abscohs = sorted(list(set(abscohs)))

    for (choice, coh, context), r in condition_averaged.items():
        if context != 'm':
            continue

        ls = get_linestyle(choice)

        idx = abscohs.index(abs(coh))
        basecolor = 'k'
        if idx == 0:
            color = apply_alpha(basecolor, 0.4)
        elif idx == 1:
            color = apply_alpha(basecolor, 0.7)
        else:
            color = apply_alpha(basecolor, 1)

        plot.plot(t[w], r[unit,w], ls, color=color, lw=lw)
        yall.append(r[unit,w])
    plot.xlim(t[w][0], t[w][-1])
    plot.xticks([t[w][0], 0, t[w][-1]])

    #-------------------------------------------------------------------------------------
    # Colour & choice
    #-------------------------------------------------------------------------------------

    plot = plots['colour_choice']
    condition_averaged = sorted_trials['colour_choice']

    abscohs = []
    for (choice, coh, context) in condition_averaged:
        abscohs.append(abs(coh))
    abscohs = sorted(list(set(abscohs)))

    for (choice, coh, context), r in condition_averaged.items():
        if context != 'c':
            continue

        ls = get_linestyle(choice)

        idx = abscohs.index(abs(coh))
        basecolor = Figure.colors('darkblue')
        if idx == 0:
            color = apply_alpha(basecolor, 0.4)
        elif idx == 1:
            color = apply_alpha(basecolor, 0.7)
        else:
            color = apply_alpha(basecolor, 1)

        plot.plot(t[w], r[unit,w], ls, color=color, lw=lw)
        yall.append(r[unit,w])
    plot.xlim(t[w][0], t[w][-1])
    plot.xticks([t[w][0], 0, t[w][-1]])

    #-------------------------------------------------------------------------------------
    # Context & choice
    #-------------------------------------------------------------------------------------

    plot = plots['context_choice']
    condition_averaged = sorted_trials['context_choice']

    for (choice, context), r in condition_averaged.items():
        ls = get_linestyle(choice)

        if context == 'm':
            color = 'k'
        else:
            color = Figure.colors('darkblue')

        plot.plot(t[w], r[unit,w], ls, color=color, lw=lw)
        yall.append(r[unit,w])
    plot.xlim(t[w][0], t[w][-1])
    plot.xticks([t[w][0], 0, t[w][-1]])

    return yall

#=========================================================================================

# Regression coefficients
CHOICE         = 0
MOTION         = 1
COLOUR         = 2
CONTEXT        = 3
CONSTANT       = 4
CHOICE_MOTION  = 5
CHOICE_COLOUR  = 6
CHOICE_CONTEXT = 7
MOTION_COLOUR  = 8
MOTION_CONTEXT = 9
COLOUR_CONTEXT = 10
nreg           = 11

def regress(trialsfile, sortedfile, betafile, dt_reg=50):
    """
    Linear regression to find task axes.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get info from first trial
    trial = trials[0]
    t     = trial['t']
    dt    = t[1] - t[0]
    step  = int(dt_reg/dt)

    #-------------------------------------------------------------------------------------
    # Setup
    #-------------------------------------------------------------------------------------

    # Consider only active units
    units = get_active_units(trialsfile)
    print("[ {}.regress ] Performing regression on {} active units."
          .format(THIS, len(units)))

    # Get preferred targets before we mess with trials
    preferred_targets = get_preferred_targets(trials)[units]

    # Stimulus period
    start, end = trials[0]['info']['epochs']['stimulus']
    t  = trials[0]['t']
    w, = np.where((start < t) & (t <= end))

    cohs_m = []
    cohs_c = []
    for trial in trials:
        cohs_m.append(trial['info']['coh_m'])
        cohs_c.append(trial['info']['coh_c'])
        trial['target'] = +1 if get_choice(trial) == 0 else -1
        trial['t']      = trial['t'][w][::step]
        trial['r']      = trial['r'][units,:][:,w][:,::step]
        trial['z']      = trial['z'][:,w][:,::step]
    maxcoh_m = max(cohs_m)
    maxcoh_c = max(cohs_c)

    #-------------------------------------------------------------------------------------
    # Normalize
    #-------------------------------------------------------------------------------------

    X  = 0
    X2 = 0
    n  = 0
    for trial in trials:
        r   = trial['r']
        X  += np.sum(r,    axis=1)
        X2 += np.sum(r**2, axis=1)
        n  += r.shape[1]
    mean = X/n
    std  = np.sqrt(X2/n - mean**2)

    mean = np.tile(mean, (r.shape[1], 1)).T
    std  = np.tile(std,  (r.shape[1], 1)).T
    for trial in trials:
        trial['r'] = (trial['r'] - mean)/std

    #-------------------------------------------------------------------------------------
    # Regress
    #-------------------------------------------------------------------------------------

    nunits, ntime = trials[0]['r'].shape

    # Coefficient matrix
    r = np.zeros((nunits, ntime, ntrials))
    F = np.zeros((nunits, nreg, ntrials))
    for i, trial in enumerate(trials):
        info = trial['info']

        # First-order terms
        r[:,:,i]       = trial['r']
        F[:,CHOICE,i]  = preferred_targets*trial['target']
        F[:,MOTION,i]  = preferred_targets*info['left_right_m']*info['coh_m']/maxcoh_m
        F[:,COLOUR,i]  = preferred_targets*info['left_right_c']*info['coh_c']/maxcoh_c
        F[:,CONTEXT,i] = +1 if info['context'] == 'm' else -1

        # Interaction terms
        F[:,CHOICE_MOTION, i] = F[:,CHOICE,i]*F[:,MOTION,i]
        F[:,CHOICE_COLOUR, i] = F[:,CHOICE,i]*F[:,COLOUR,i]
        F[:,CHOICE_CONTEXT,i] = F[:,CHOICE,i]*F[:,CONTEXT,i]
        F[:,MOTION_COLOUR, i] = F[:,MOTION,i]*F[:,COLOUR,i]
        F[:,MOTION_CONTEXT,i] = F[:,MOTION,i]*F[:,CONTEXT,i]
        F[:,COLOUR_CONTEXT,i] = F[:,COLOUR,i]*F[:,CONTEXT,i]
    F[:,CONSTANT,:] = 1

    # Regression coefficients
    beta = np.zeros((nunits, ntime, nreg))
    for i in xrange(nunits):
        A = np.linalg.inv(F[i].dot(F[i].T)).dot(F[i])
        for k in xrange(ntime):
            beta[i,k] = A.dot(r[i,k])
            if np.any(np.isnan(beta[i,k])):
                raise RuntimeError("[ {}.regress ] Regression failed.".format(THIS))

    #-------------------------------------------------------------------------------------
    # Denoising matrix
    #-------------------------------------------------------------------------------------

    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    all_conditions = sorted_trials['all']
    for cond, r in all_conditions.items():
        all_conditions[cond] = r[units,::step]

    # Data matrix
    X = np.zeros((all_conditions.values()[0].shape[0],
                  len(all_conditions)*all_conditions.values()[0].shape[1]))
    c = 0
    for cond, r in sorted_trials['all'].items():
        X[:,c:c+r.shape[1]] = r
        c += r.shape[1]

    U, S, V = np.linalg.svd(X.T)
    assert np.all(S[:-1] >= S[1:])

    npca = 12
    W    = V[:npca,:]
    D    = (W.T).dot(W)
    assert np.all(D.T == D)

    #-------------------------------------------------------------------------------------
    # Task axes
    #-------------------------------------------------------------------------------------

    # Rearrange from (units, time, reg) to (reg, time, units)
    beta = np.swapaxes(beta, 0, 2)

    # Denoise
    beta = beta.dot(D.T)

    # Time-independent regression vectors
    beta_max = np.zeros((nreg, nunits))
    for v in xrange(nreg):
        imax        = np.argmax(np.linalg.norm(beta[v], axis=1))
        beta_max[v] = beta[v,imax]

    Bmax = beta_max[:4].T
    Q, R = np.linalg.qr(Bmax)
    Q    = Q*np.sign(np.diag(R))

    #-------------------------------------------------------------------------------------
    # Save
    #-------------------------------------------------------------------------------------

    with open(betafile, 'wb') as f:
        pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
    print("[ {}.regress ] Regression coefficients saved to {}".format(THIS, betafile))

def plot_regress(betafile, plots):
    # Regression coefficients
    with open(betafile) as f:
        beta = pickle.load(f)

    regaxes = {'choice': CHOICE, 'motion': MOTION, 'colour': COLOUR, 'context': CONTEXT}
    for k, plot in plots.items():
        Y, X = k.split('_')

        plot.equal()

        # Annoying result of copying the Mante paper
        if X == 'colour':
            s = 'color'
        else:
            s = X
        plot.xlabel(s.capitalize())

        if Y == 'colour':
            s = 'color'
        else:
            s = Y
        plot.ylabel(s.capitalize())

        x = beta[:,regaxes[X]]
        y = beta[:,regaxes[Y]]
        plot.plot(x, y, 'o', mfc='0.2', mec='w', ms=2.5, mew=0.3, zorder=10)

        M = 0.3
        #assert np.all(abs(x) <= M)
        #assert np.all(abs(y) <= M)

        plot.xlim(-M, M)
        plot.xticks([-M, 0, M])

        plot.ylim(-M, M)
        plot.yticks([-M, 0, M])

        plot.hline(0, lw=0.75, color='k', zorder=1)
        plot.vline(0, lw=0.75, color='k', zorder=1)

#=========================================================================================
# State space
#=========================================================================================

def plot_taskaxes(plot, yax, p_vc, basecolor):
    abscohs = []
    for choice, coh, context in p_vc:
        abscohs.append(abs(coh))
    abscohs = sorted(list(set(abscohs)))

    #-------------------------------------------------------------------------------------
    # Subtract mean
    #-------------------------------------------------------------------------------------

    p = p_vc.values()[0]
    Xchoice = np.zeros_like(p[CHOICE])
    Xmotion = np.zeros_like(p[MOTION])
    Xcolour = np.zeros_like(p[COLOUR])

    for p in p_vc.values():
        Xchoice += p[CHOICE]
        Xmotion += p[MOTION]
        Xcolour += p[COLOUR]
    mean_choice = Xchoice/len(p_vc)
    mean_motion = Xmotion/len(p_vc)
    mean_colour = Xcolour/len(p_vc)

    for cond, p in p_vc.items():
        p[CHOICE] -= mean_choice
        p[MOTION] -= mean_motion
        p[COLOUR] -= mean_colour

    #-------------------------------------------------------------------------------------

    xall = []
    yall = []
    for cond, p in p_vc.items():
        idx = abscohs.index(abs(cond[1]))
        if idx == 0:
            color = apply_alpha(basecolor, 0.4)
        elif idx == 1:
            color = apply_alpha(basecolor, 0.7)
        else:
            color = apply_alpha(basecolor, 1)

        if cond[1] > 0:
            prop = dict(mfc=color, mec=color, ms=2.5, mew=0.5)
        else:
            prop = dict(mfc='none', mec=color, ms=3, mew=0.5)

        plot.plot(p[CHOICE], p[yax], '-', color=color, lw=0.75)
        plot.plot(p[CHOICE], p[yax], 'o', color=color, **prop)

        xall.append(p[CHOICE])
        yall.append(p[yax])

    if yax == MOTION:
        plot.ylabel('Motion')
    elif yax == COLOUR:
        plot.ylabel('Color')

    return np.concatenate(xall), np.concatenate(yall)

def plot_statespace(trialsfile, sortedfile, betafile, plots):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    # Load task axes
    with open(betafile) as f:
        M = pickle.load(f).T

    # Active units
    units = get_active_units(trialsfile)

    # Epoch to plot
    start, end = trials[0]['info']['epochs']['stimulus']
    start += 0
    end   += 0
    w, = np.where((start <= t) & (t <= end))

    # Down-sample
    dt   = t[1] - t[0]
    step = int(50/dt)
    w    = w[::step]

    # Colors
    color_m = 'k'
    color_c = Figure.colors('darkblue')

    xall = []
    yall = []

    #-------------------------------------------------------------------------------------
    # Labels
    #-------------------------------------------------------------------------------------

    plots['c1'].xlabel('Choice')

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by coherence
    #-------------------------------------------------------------------------------------

    plot = plots['m1']

    p_vc = {}
    for cond, r in sorted_trials['motion_choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    plot.ylabel('Motion')

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by coherence
    #-------------------------------------------------------------------------------------

    plot = plots['m2']
    p_vc = {}
    for cond, r in sorted_trials['motion_choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Motion context: colour vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['m3']
    p_vc = {}
    for cond, r in sorted_trials['colour_choice'].items():
        if cond[2] == 'm':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------

    plot = plots['c1']
    p_vc = {}
    for cond, r in sorted_trials['motion_choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_m)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: motion vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['c2']
    p_vc = {}
    for cond, r in sorted_trials['colour_choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, MOTION, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Colour context: colour vs. choice, sorted by colour
    #-------------------------------------------------------------------------------------

    plot = plots['c3']
    p_vc = {}
    for cond, r in sorted_trials['colour_choice'].items():
        if cond[2] == 'c':
            p_vc[cond] = M.dot(r[units,:][:,w])
    x, y = plot_taskaxes(plot, COLOUR, p_vc, color_c)
    xall.append(x)
    yall.append(y)

    #-------------------------------------------------------------------------------------
    # Shared axes
    #-------------------------------------------------------------------------------------

    xall = np.concatenate(xall)
    yall = np.concatenate(yall)

    for plot in plots.values():
        plot.aspect(1.5)
        plot.lim('x', xall)
        plot.lim('y', yall)

#=========================================================================================
# Task manager
#=========================================================================================

def do(action, args, p):
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #-------------------------------------------------------------------------------------
    # Trials
    #-------------------------------------------------------------------------------------

    if action == 'trials':
        run_trials(p, args)

    #-------------------------------------------------------------------------------------
    # Psychometric function
    #-------------------------------------------------------------------------------------

    elif action == 'psychometric':
        #---------------------------------------------------------------------------------
        # Figure setup
        #---------------------------------------------------------------------------------

        w   = 6.5
        h   = 3
        fig = Figure(w=w, h=h, axislabelsize=7.5, labelpadx=6, labelpady=7.5,
                     thickness=0.6, ticksize=3, ticklabelsize=6.5, ticklabelpad=2)

        w = 0.39
        h = 0.7

        L = 0.09
        R = L + w + 0.1
        y = 0.2

        plots = {'m': fig.add([L, y, w, h]),
                 'c': fig.add([R, y, w, h])}

        #---------------------------------------------------------------------------------
        # Labels
        #---------------------------------------------------------------------------------

        plot = plots['m']
        plot.xlabel('Motion coherence (\%)')
        plot.ylabel('Choice to right (\%)')

        plot = plots['c']
        plot.xlabel('Color coherence (\%)')
        plot.ylabel('Choice to green (\%)')

        #---------------------------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------------------------

        trialsfile = get_trialsfile(p)
        psychometric_function(trialsfile, plots)

        # Legend
        prop = {'prop': {'size': 7}, 'handlelength': 1.2,
                'handletextpad': 1.1, 'labelspacing': 0.5}
        plots['m'].legend(bbox_to_anchor=(0.41, 1), **prop)

        #---------------------------------------------------------------------------------

        fig.save(path=p['figspath'], name=p['name']+'_'+action)
        fig.close()

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    elif action == 'sort':
        trialsfile = get_trialsfile(p)
        sortedfile = get_sortedfile(p)
        sort_trials(trialsfile, sortedfile)

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity
    #-------------------------------------------------------------------------------------

    elif action == 'units':
        from glob import glob

        # Remove existing files
        print("[ {}.do ]".format(THIS))
        filenames = glob('{}_unit*'.format(join(p['figspath'], p['name'])))
        for filename in filenames:
            os.remove(filename)
            print("  Removed {}".format(filename))

        trialsfile = get_trialsfile(p)
        sortedfile = get_sortedfile(p)

        units = get_active_units(trialsfile)
        for unit in units:
            #-----------------------------------------------------------------------------
            # Figure setup
            #-----------------------------------------------------------------------------

            w   = 2.5
            h   = 6
            fig = Figure(w=w, h=h, axislabelsize=7.5, labelpadx=6, labelpady=7.5,
                         thickness=0.6, ticksize=3, ticklabelsize=6.5, ticklabelpad=2)

            w  = 0.55
            x0 = 0.3

            h  = 0.17
            dy = h + 0.06
            y0 = 0.77
            y1 = y0 - dy
            y2 = y1 - dy
            y3 = y2 - dy

            plots = {
                'choice':         fig.add([x0, y0, w, h]),
                'motion_choice':  fig.add([x0, y1, w, h]),
                'colour_choice':  fig.add([x0, y2, w, h]),
                'context_choice': fig.add([x0, y3, w, h])
                }

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(unit, sortedfile, plots, sortby_fontsize=7)
            plots['context_choice'].xlabel('Time (ms)')

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'], name='{}_unit{}'.format(p['name'], unit))
            fig.close()
        print("[ {}.do ] {} units processed.".format(THIS, len(units)))

    #-------------------------------------------------------------------------------------
    # Regress
    #-------------------------------------------------------------------------------------

    elif action == 'regress':
        trialsfile = get_trialsfile(p)
        sortedfile = get_sortedfile(p)
        betafile   = get_betafile(p)
        regress(trialsfile, sortedfile, betafile)

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action.".format(THIS))

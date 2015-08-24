"""
Analyze the Romo task.

"""
from __future__ import division

import cPickle as pickle
import os
import sys

import numpy       as np
import scipy.stats as stats

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure, mpl

#=========================================================================================
# Setup
#=========================================================================================

# For messages
THIS = "examples.analysis.romo"

# File to store trials in
def get_trialsfile(p):
    return '{}/{}_trials.pkl'.format(p['trialspath'], p['name'])

# Load trials
def load_trials(trialsfile):
    with open(trialsfile) as f:
        trials = pickle.load(f)

    return trials, len(trials)

# File to store sorted trials in
def get_sortedfile(p):
    return '{}/{}_sorted.pkl'.format(p['trialspath'], p['name'])

# File to store d'
def get_dprimefile(p):
    return '{}/{}_dprime.txt'.format(p['datapath'], p['name'])

# File to store selectivity
def get_selectivityfile(p):
    return '{}/{}_selectivity.txt'.format(p['datapath'], p['name'])

# Simple choice function
def get_choice(trial):
    return np.argmax(trial['z'][:,-1])

def safe_divide(x):
    if x == 0:
        return 0
    return 1/x

# Nice colors to represent coherences
colors = {
    18: '#9ecae1',
    22: '#6baed6',
    26: '#4292c6',
    30: '#2171b5',
    34: '#084594',
    18: '#9ecae1',
    22: '#6baed6',
    26: '#4292c6',
    30: '#2171b5',
    34: '#084594'
    }

# Integration threshold
threshold = 0.8

#=========================================================================================

def run_trials(p, args):
    """
    Run trials.

    """
    # Model
    m = p['model']

    # Number of trials
    try:
        ntrials = int(args[0])
    except:
        ntrials = 100*m.nconditions

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            # Condition
            b      = i % m.nconditions
            k0, k1 = tasktools.unravel_index(b, (len(m.fpairs), len(m.gt_lts)))
            fpair  = m.fpairs[k0]
            gt_lt  = m.gt_lts[k1]

            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':  'test',
                'catch': False, 
                'fpair': fpair,
                'gt_lt': gt_lt
                }
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            if info['f1'] > info['f2']:
                gt_lt = '>'
            else:
                gt_lt = '<'
            s = ("Trial {:>{}}/{}: {:>2} {} {:>2}"
                 .format(i+1, w, ntrials, info['f1'], gt_lt, info['f2']))
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
                'info': info
                }
            trials.append(trial)
    except KeyboardInterrupt:
        pass
    print("")

    # Save all
    filename = get_trialsfile(p)
    with open(filename, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)
    print("[ {}.run_trials ] Trials saved to {}".format(THIS, filename))

    # Psychometric function
    psychometric_function(filename)

#=========================================================================================

def psychometric_function(trialsfile, plot=None, smap=None, **kwargs):
    """
    Compute and plot the sychometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric function
    #-------------------------------------------------------------------------------------

    correct_by_fpair = {}
    ncorrect = 0
    for trial in trials:
        info = trial['info']
        
        # Signed coherence
        f1 = info['f1']
        f2 = info['f2']

        # Correct
        choice = get_choice(trial)
        correct_by_fpair.setdefault((f1, f2), []).append(1*(choice == info['choice']))

        # Overall correct
        if choice == info['choice']:
            ncorrect += 1
    for fpair, correct in correct_by_fpair.items():
        correct_by_fpair[fpair] = sum(correct)/len(correct)

    # Report overall performance
    pcorrect = 100*ncorrect/ntrials
    print("[ {}.psychometric_function ] {:.2f}% correct.".format(THIS, pcorrect))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is not None:
        plot.equal()

        xall = []
        yall = []
        for (f1, f2), correct in correct_by_fpair.items():
            if smap is not None:
                plot.circle((f1, f2), 2.1, ec='none', fc=smap.to_rgba(f1))
                color = 'w'
            else:
                color = 'k'
            plot.text(f1, f2, '{}'.format(int(100*correct)), ha='center', va='center',
                      fontsize=kwargs.get('fontsize', 8), color=color)
            xall.append(f1)
            yall.append(f2)

        plot.lim('x', xall, margin=0.13)
        plot.lim('y', yall, margin=0.13)

        # Equality line
        xmin, xmax = plot.get_xlim()
        ymin, ymax = plot.get_ylim()
        plot.plot([xmin, xmax], [ymin, ymax], color='k', lw=kwargs.get('lw', 1))

#=========================================================================================

def sort_trials(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get unique conditions
    conds = list(set([trial['info']['f1'] for trial in trials]))

    #-------------------------------------------------------------------------------------
    # Average across conditions
    #-------------------------------------------------------------------------------------

    nunits, ntime = trials[0]['r'].shape
    t = trials[0]['t']

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: np.zeros(ntime)           for c in conds}
    for trial in trials:
        info   = trial['info']
        choice = get_choice(trial)

        # Include only correct trials
        if choice != info['choice']:
            continue

        # Sort by f1
        f1 = info['f1']
        sorted_trials[f1]   += trial['r']
        ntrials_by_cond[f1] += 1
    for c in conds:
        sorted_trials[c] /= ntrials_by_cond[c]

    # Save
    with open(sortedfile, 'wb') as f:
        pickle.dump((t, sorted_trials), f, pickle.HIGHEST_PROTOCOL)
        print("[ {}.sort_trials ] Sorted trials saved to {}".format(THIS, sortedfile))

def plot_unit(unit, sortedfile, plot, smap, t0=0, tmin=-np.inf, tmax=np.inf, **kwargs):
    """
    Plot single-unit activity sorted by condition.

    unit : int
    sortedfile : str
    plot : pycog.figtools.Subplot
    
    """
    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    # Time
    w, = np.where((tmin <= t) & (t <= tmax))
    t  = t[w] - t0
    t  = 1e-3*t

    conds = sorted_trials.keys()

    all = []
    for f1 in sorted(conds, key=lambda f1: f1):
        prop = {'color': smap.to_rgba(f1),
                'lw':    kwargs.get('lw', 1.5),
                'label': '{} Hz'.format(f1)}

        r = sorted_trials[f1][unit][w]
        plot.plot(t, r, **prop)
        all.append(r)

    plot.xlim(t[0], t[-1])
    plot.lim('y', all, lower=0)

    return np.concatenate(all)

#=========================================================================================

def get_active_units(sorted_trials):
    """
    Get units that do something.

    """
    N = next(sorted_trials.itervalues()).shape[0]
    units = []
    for i in xrange(N):
        active = False
        for r in sorted_trials.values():
            if np.std(r[i]) > 0.1:
                active = True
                break
        if active:
            units.append(i)

    return units

def tuning(sortedfile):
    """
    Fit r = a0 + a1*f1.

    """
    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    # Get f1s
    f1s = sorted(sorted_trials.keys())

    # Active units
    units = get_active_units(sorted_trials)
    units = range(next(sorted_trials.itervalues()).shape[0])

    # Gather data in convenient form
    data = np.zeros((len(units), len(t), len(f1s)))
    for k, f1 in enumerate(f1s):
        data[:,:,k] = sorted_trials[f1][units,:]

    # Regress
    a1s   = np.zeros((len(units), len(t)))
    pvals = np.zeros_like(a1s)
    for i in xrange(len(units)):
        for j in xrange(len(t)):
            slope, intercept, rval, pval, stderr = stats.linregress(f1s, data[i,j])
            a1s[i,j]   = slope
            pvals[i,j] = pval

    return units, a1s, pvals

def tuning_corr(trialsfile, sortedfile, plot_sig, plot_corr=None, 
                plot_stim=None, plot_delay=None, t0=0, **kwargs):
    """
    Plot correlation of a1 between different times.

    """
    units, a1s, pvals = tuning(sortedfile)

    # Get first trial
    trials, _ = load_trials(trialsfile)
    trial = trials[0]
    info  = trial['info']
    t     = trial['t']

    _, delay_end = info['epochs']['delay']
    delay_end = 1e-3*(delay_end - t0)

    t_stim   = np.mean(info['epochs']['f1'])
    idx_stim = np.where(t >= t_stim)[0][0]

    t_delay   = np.mean(info['epochs']['delay'])
    idx_delay = np.where(t >= t_delay)[0][0]
    
    t_delay_end   = info['epochs']['delay'][1]
    idx_delay_end = np.where(t > t_delay_end)[0][0]

    t_f2_end   = info['epochs']['f2'][1]
    idx_f2_end = np.where(t > t_f2_end)[0][0]

    t_all   = 1e-3*(t[idx_stim:idx_delay_end]-t0)
    idx_all = np.arange(len(t))[idx_stim:idx_delay_end]

    # Plot correlation across time
    if plot_corr is not None:
        plot = plot_corr

        # With stimulus period
        corr = []
        for k in idx_all:
            corr.append(stats.pearsonr(a1s[:,idx_stim], a1s[:,k])[0])
        plot.plot(t_all, corr, color=Figure.colors('blue'), lw=kwargs.get('lw', 1))

        # With stimulus period
        corr = []
        for k in idx_all:
            corr.append(stats.pearsonr(a1s[:,idx_delay], a1s[:,k])[0])
        plot.plot(t_all, corr, color=Figure.colors('green'), lw=kwargs.get('lw', 1))

        plot.xlim(-1e-3*t0, delay_end)
        plot.ylim(-1, 1)

    # Plot fraction of significantly tuned units.
    if plot_sig is not None:
        plot = plot_sig

        psig = np.sum(1*(pvals < 0.05), axis=0)/len(units)
        plot.plot(1e-3*(t[1:idx_f2_end]-t0), psig[1:idx_f2_end],
                  color='0.2', lw=kwargs.get('lw', 1))

        plot.xlim(1e-3*(t[0]-t0), 1e-3*(t[idx_f2_end-1]-t0))
        #plot.lim('y', psig, lower=0)
        plot.ylim(0, 1)

    # Shared plot properties
    prop = {'mfc': '0.2',
            'mec': 'none',
            'ms':  kwargs.get('ms', 2)}

    # Plot a1, end of delay vs. stimulus
    if plot_stim is not None:
        plot = plot_stim
        plot.equal()

        for i in xrange(len(units)):
            plot.plot(a1s[i,idx_stim], a1s[i,idx_delay_end], 'o', **prop)

    # Plot a1, end of delay vs. middle of delay
    if plot_delay is not None:
        plot = plot_delay
        plot.equal()

        for i in xrange(len(units)):
            plot.plot(a1s[i,idx_delay], a1s[i,idx_delay_end], 'o', **prop)

#=========================================================================================

def do(action, args, p):
    """
    Manage tasks.

    """
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
        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------------------------

        trialsfile = get_trialsfile(p)
        psychometric_function(trialsfile, plot)

        plot.xlabel('$f_1$ (Hz)')
        plot.ylabel('$f_2$ (Hz)')

        #---------------------------------------------------------------------------------

        fig.save(path=p['figspath'], name=p['name']+'_'+action)
        fig.close()

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    elif action == 'sort':
        sort_trials(get_trialsfile(p), get_sortedfile(p))

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to stimulus onset
    #-------------------------------------------------------------------------------------

    elif action == 'units':
        from glob import glob

        # Remove existing files
        print("[ {}.do ]".format(THIS))
        filenames = glob('{}/{}_unit*'.format(p['figspath'], p['name']))
        for filename in filenames:
            os.remove(filename)
            print("  Removed {}".format(filename))

        # Load sorted trials
        sortedfile = get_sortedfile(p)
        with open(sortedfile) as f:
            t, sorted_trials = pickle.load(f)

        # Colormap
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=p['model'].fmin, vmax=p['model'].fmax)
        smap = mpl.cm.ScalarMappable(norm, cmap)

        for i in xrange(p['model'].N):
            # Check if the unit does anything
            active = False
            for condition_averaged in sorted_trials.values():
                if np.std(condition_averaged[i]) > 0.1:
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot, smap)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'], name=p['name']+'_unit{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Selectivity
    #-------------------------------------------------------------------------------------

    elif action == 'selectivity':
        # Model
        m = p['model']
        
        trialsfile = get_trialsfile(p)
        dprime     = get_choice_selectivity(trialsfile)

        def get_first(x, p):
            return x[:int(p*len(x))]

        psig  = 0.25
        units = np.arange(len(dprime))
        try:
            idx = np.argsort(abs(dprime[m.EXC]))[::-1]
            exc = get_first(units[m.EXC][idx], psig)

            idx = np.argsort(abs(dprime[m.INH]))[::-1]
            inh = get_first(units[m.INH][idx], psig)

            idx = np.argsort(dprime[exc])[::-1]
            units_exc = list(exc[idx])

            idx = np.argsort(dprime[inh])[::-1]
            units_inh = list(units[inh][idx])

            units  = units_exc + units_inh
            dprime = dprime[units]
        except AttributeError:
            idx = np.argsort(abs(dprime))[::-1]
            all = get_first(units[idx], psig)

            idx    = np.argsort(dprime[all])[::-1]
            units  = list(units[all][idx])
            dprime = dprime[units]

        # Save d'
        filename = get_dprimefile(p)
        np.savetxt(filename, dprime)
        print("[ {}.do ] d\' saved to {}".format(THIS, filename))

        # Save selectivity
        filename = get_selectivityfile(p)
        np.savetxt(filename, units, fmt='%d')
        print("[ {}.do ] Choice selectivity saved to {}".format(THIS, filename))

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action '{}'.".format(THIS, action))

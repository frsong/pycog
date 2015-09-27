"""
Analyze variants of the multisensory integration task.

"""
from __future__ import division

import cPickle as pickle
import os
import sys
from   os.path import join

import numpy as np

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure

THIS = "examples.analysis.multisensory"

#=========================================================================================
# Setup
#=========================================================================================

# File to store trials in
def get_trialsfile(p):
    return join(p['trialspath'], p['name'] + '_trials.pkl')

# File to store sorted trials in
def get_sortedfile(p):
    return join(p['trialspath'], p['name'] + '_sorted.pkl')

# Simple choice function
def get_choice(trial):
    return np.argmax(trial['z'][:,-1])

# Define "active" units
def is_active(r):
    return np.std(r) > 0.05

# Colors
colors = {
    'v':  Figure.colors('blue'),
    'a':  Figure.colors('green'),
    'va': Figure.colors('orange')
    }

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
        ntrials = 100
    ntrials *= m.nconditions

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            b        = i % m.nconditions
            k1, k2   = tasktools.unravel_index(b, (len(m.modalities), len(m.freqs)))
            modality = m.modalities[k1]
            freq     = m.freqs[k2]

            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':     'test',
                'catch':    False,
                'modality': modality,
                'freq':     freq
                }
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            if info['modality'] == 'v':
                s = "Trial {:>{}}/{}: v |{:>2}".format(i+1, w, ntrials, info['freq'])
            elif info['modality'] == 'a':
                s = "Trial {:>{}}/{}:  a|{:>2}".format(i+1, w, ntrials, info['freq'])
            else:
                s = "Trial {:>{}}/{}: va|{:>2}".format(i+1, w, ntrials, info['freq'])
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
    size = os.path.getsize(filename)*1e-9
    print("[ {}.run_trials ] Trials saved to {} ({:.1f} GB)".format(THIS, filename, size))

    # Compute the psychometric function
    psychometric_function(filename)

#=========================================================================================

def psychometric_function(trialsfile, plot=None, **kwargs):
    """
    Compute and plot the sychometric function.

    """
    # Load trials
    with open(trialsfile) as f:
        trials = pickle.load(f)
    ntrials = len(trials)

    # Get modalities
    mods = list(set([trial['info']['modality'] for trial in trials]))

    #-------------------------------------------------------------------------------------
    # Compute psychometric function
    #-------------------------------------------------------------------------------------

    results         = {mod: {} for mod in mods}
    ncorrect_by_mod = {mod: 0  for mod in mods}
    ntrials_by_mod  = {mod: 0  for mod in mods}
    for trial in trials:
        # Condition
        info = trial['info']
        mod  = info['modality']
        freq = info['freq']
        ntrials_by_mod[mod] += 1

        # Choice
        choice = get_choice(trial)
        results[mod].setdefault(freq, []).append(choice)

        # Correct
        if choice == info['choice']:
            ncorrect_by_mod[mod] += 1

    # Report overall performance
    pcorrect_by_mod = {mod: 100*ncorrect_by_mod[mod]/ntrials_by_mod[mod] for mod in mods}
    print("[ {}.psychometric_function ]".format(THIS))
    print("  v  {:.2f}% correct.".format(pcorrect_by_mod['v']))
    print("   a {:.2f}% correct.".format(pcorrect_by_mod['a']))
    print("  va {:.2f}% correct.".format(pcorrect_by_mod['va']))

    # Psychometric function
    for mod in mods:
        choice_by_freq = results[mod]

        freqs = np.sort(np.array(choice_by_freq.keys()))
        p0    = np.zeros(len(freqs))
        for i, freq in enumerate(freqs):
            choices = np.array(choice_by_freq[freq])
            p0[i]   = 1 - np.sum(choices)/len(choices)

        results[mod] = (freqs, p0)

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is not None:
        sigmas = {}
        for mod in ['v', 'a', 'va']:
            freqs, p0 = results[mod]

            # Data
            prop = {'ms':     kwargs.get('ms',  6),
                    'mfc':    kwargs.get('mfc', colors[mod]),
                    'mew':    kwargs.get('mew', 0),
                    'zorder': 10}
            plot.plot(freqs, 100*p0, 'o', **prop)

            if mod == 'v':
                label = 'Visual'
            elif mod == 'a':
                label = 'Auditory'
            elif mod == 'va':
                label = 'Multisensory'
            else:
                raise ValueError(" [ {}.psychometric_function ] Unknown modality.".
                                 format(THIS))

            # Fit
            prop = {'color':  kwargs.get('mfc', colors[mod]),
                    'lw':     kwargs.get('lw',  1),
                    'label':  label,
                    'zorder': 5}
            try:
                popt, func = fittools.fit_psychometric(freqs, p0)

                fit_freqs = np.linspace(min(freqs), max(freqs), 201)
                fit_p0    = func(fit_freqs, **popt)
                plot.plot(fit_freqs, 100*fit_p0, **prop)

                sigmas[mod] = popt['sigma']
            except RuntimeError:
                print("[ {}.psychometric_function ]".format(THIS)
                      + " Unable to fit, drawing a line through the points.")
                plot.plot(freqs, 100*p0, **prop)

        plot.ylim(0, 100)
        plot.yticks([0, 25, 50, 75, 100])
        plot.lim('x', freqs)

        # Is it optimal?
        print("")
        print("  Optimality test")
        print("  ---------------")
        for mod in ['v', 'a', 'va']:
            print("  sigma_{:<2} = {:.6f}".format(mod, sigmas[mod]))
        print("  1/sigma_v**2 + 1/sigma_a**2 = {:.6f}"
              .format(1/sigmas['v']**2 + 1/sigmas['a']**2))
        print("  1/sigma_va**2               = {:.6f}".format(1/sigmas['va']**2))

#=========================================================================================

def sort_trials(trialsfile, sortedfile):
    # Load trials
    with open(trialsfile) as f:
        trials = pickle.load(f)
    ntrials = len(trials)

    # Get unique conditions
    conds = list(set([(trial['info']['modality'], trial['info']['choice'])
                      for trial in trials if len(trial['info']['modality']) == 1]))

    nunits, ntime = trials[0]['r'].shape
    t = trials[0]['t']

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: 0 for c in conds}
    for trial in trials:
        info   = trial['info']
        mod    = info['modality']
        choice = get_choice(trial)

        # Correct trials only
        if choice == info['choice']:
            c = (mod, choice)
            if c in sorted_trials:
                sorted_trials[c]   += trial['r']
                ntrials_by_cond[c] += 1
    for c in conds:
        sorted_trials[c] /= ntrials_by_cond[c]

    #-------------------------------------------------------------------------------------
    # Save
    #-------------------------------------------------------------------------------------

    with open(sortedfile, 'wb') as f:
        pickle.dump((t, sorted_trials), f, pickle.HIGHEST_PROTOCOL)
    print("[ {}.sort_trials ] Sorted trials saved to {}".format(THIS, sortedfile))

def plot_unit(unit, sortedfile, plot, t0=0, tmin=-np.inf, tmax=np.inf, **kwargs):
    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    # Time
    w, = np.where((tmin <= t) & (t <= tmax))
    t  = t[w] - t0

    all = []
    for c in [('v', 0), ('v', 1), ('a', 0), ('a', 1)]:
        mod, choice = c

        if mod == 'v':
            label = "Vis, "
        else:
            label = "Aud, "

        prop = {'color': colors[mod],
                'lw':    kwargs.get('lw', 1.5)}
        if choice == 1:
            label += "low"
            prop['ls'] = '-'
        else:
            label += "high"
            prop['ls']     = '--'
            prop['dashes'] = kwargs.get('dashes', [3.5, 2.5])

        r = sorted_trials[c][unit][w]
        plot.plot(t, r, label=label, **prop)
        all.append(r)

    plot.xlim(t[0], t[-1])
    plot.xticks([t[0], 0, t[-1]])
    plot.lim('y', all, lower=0)

    return np.concatenate(all)

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

        plot.xlabel(r'Rate (events/sec)')
        plot.ylabel(r'Percent high')

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

        # Load sorted trials
        sortedfile = get_sortedfile(p)
        with open(sortedfile) as f:
            t, sorted_trials = pickle.load(f)

        for i in xrange(p['model'].N):
            # Check if the unit does anything
            active = False
            for r in sorted_trials.values():
                if is_active(r[i]):
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #---------------------------------------------------------------------------------
            # Plot
            #---------------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            #---------------------------------------------------------------------------------

            fig.save(path=p['figspath'], name=p['name']+'_unit{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action.".format(THIS))

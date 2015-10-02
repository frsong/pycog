"""
Analyze variants of the random dots motion task.

"""
from __future__ import division

import cPickle as pickle
import os
import sys
from   os.path import join

import numpy as np

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure

THIS = "examples.analysis.rdm"

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
def get_sortedfile_stim_onset(p):
    return join(p['trialspath'], p['name'] + '_sorted_stim_onset.pkl')

# File to store sorted trials in
def get_sortedfile_response(p):
    return join(p['trialspath'], p['name'] + '_sorted_response.pkl')

# File to store d'
def get_dprimefile(p):
    return join(p['datapath'], p['name'] + '_dprime.txt')

# File to store selectivity
def get_selectivityfile(p):
    return join(p['datapath'], p['name'] + '_selectivity.txt')

def safe_divide(x):
    if x == 0:
        return 0
    return 1/x

# Define "active" units
def is_active(r):
    return np.std(r) > 0.1

# Nice colors to represent coherences, from http://colorbrewer2.org/
colors = {
        0:  '#c6dbef',
        1:  '#9ecae1',
        2:  '#6baed6',
        4:  '#4292c6',
        8:  '#2171b5',
        16: '#084594'
        }

# Decision threshold
THRESHOLD = 1

# Coherence scale
SCALE = 3.2

# Simple choice function
def get_choice(trial, threshold=False):
    if not threshold:
        return np.argmax(trial['z'][:,-1])

    # Reaction time
    w0, = np.where(trial['z'][0] > THRESHOLD)
    w1, = np.where(trial['z'][1] > THRESHOLD)
    if len(w0) == 0 and len(w1) == 0:
        return None

    if len(w1) == 0:
        return 0, w0[0]
    if len(w0) == 0:
        return 1, w1[0]
    if w0[0] < w1[0]:
        return 0, w0[0]
    return 1, w1[0]

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
    ntrials *= m.nconditions + 1

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            b = i % (m.nconditions + 1)
            if b == 0:
                # Zero-coherence condition
                coh    = 0
                in_out = rng.choice(m.in_outs)
            else:
                # All other conditions
                k1, k2 = tasktools.unravel_index(b-1, (len(m.cohs), len(m.in_outs)))
                coh    = m.cohs[k1]
                in_out = m.in_outs[k2]

            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':   'test',
                'catch':  False,
                'coh':    coh,
                'in_out': in_out
                }
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            if coh == 0:
                s = "Trial {:>{}}/{}: {:>3}".format(i+1, w, ntrials, info['coh'])
            else:
                s = ("Trial {:>{}}/{}: {:>+3}"
                     .format(i+1, w, ntrials, info['in_out']*info['coh']))
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

def psychometric_function(trialsfile, plot=None, threshold=False, **kwargs):
    """
    Compute and plot the sychometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric function
    #-------------------------------------------------------------------------------------

    choice_by_coh = {}
    ntot     = 0
    ncorrect = 0
    for trial in trials:
        info = trial['info']

        # Signed coherence
        coh = info['in_out']*info['coh']

        # Choice
        choice = get_choice(trial, threshold)
        if choice is None:
            continue
        if coh != 0:
            ntot += 1
        if isinstance(choice, tuple):
            choice, _ = choice
        choice_by_coh.setdefault(coh, []).append(choice)

        # Correct
        if coh != 0 and choice == info['choice']:
            ncorrect += 1

    # Report overall performance
    pcorrect = 100*ncorrect/ntot
    print("[ {}.psychometric_function ] {}/{} = {:.2f}% correct."
          .format(THIS, ncorrect, ntot, pcorrect))

    cohs = np.sort(np.array(choice_by_coh.keys()))
    p0   = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        choices = np.array(choice_by_coh[coh])
        p0[i]   = 1 - np.sum(choices)/len(choices)
    scaled_cohs = SCALE*cohs

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is not None:
        # Data
        prop = {'ms':     kwargs.get('ms',  6),
                'mfc':    kwargs.get('mfc', '0.2'),
                'mew':    kwargs.get('mew', 0),
                'zorder': 10}
        plot.plot(scaled_cohs, 100*p0, 'o', **prop)

        # Fit
        prop = {'color':  kwargs.get('mfc', '0.2'),
                'lw':     kwargs.get('lw',  1),
                'zorder': 5}
        try:
            popt, func = fittools.fit_psychometric(scaled_cohs, p0)

            fit_cohs = np.linspace(min(scaled_cohs), max(scaled_cohs), 201)
            fit_p0   = func(fit_cohs, **popt)
            plot.plot(fit_cohs, 100*fit_p0, **prop)
        except RuntimeError:
            print("[ {}.psychometric_function ]".format(THIS)
                  + " Unable to fit, drawing a line through the points.")
            plot.plot(scaled_cohs, 100*p0, **prop)

        plot.ylim(0, 100)
        plot.yticks([0, 25, 50, 75, 100])
        plot.lim('x', scaled_cohs)

#=========================================================================================

def plot_stimulus_duration(trialsfile, plot, **kwargs):
    """
    Percent correct as a function of stimulus duration.

    """
    from pycog.datatools import partition

    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric performance by stimulus duration
    #-------------------------------------------------------------------------------------

    correct_duration_by_coh = {}
    for i, trial in enumerate(trials):
        info = trial['info']

        # Coherence
        coh = info['coh']
        if coh == 0:
            continue

        # Correct, stimulus duration
        correct = 1*(get_choice(trial) == info['choice'])
        correct_duration_by_coh.setdefault(coh, ([], []))[0].append(correct)
        correct_duration_by_coh[coh][1].append(np.ptp(info['epochs']['stimulus']))

        correct_by_coh = {}

    correct_by_coh = {}
    for coh, (correct, duration) in correct_duration_by_coh.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(correct),
                                            nbins=10)
        correct_by_coh[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                               [100*np.sum(Ybin > 0)*safe_divide(len(Ybin))
                                for Ybin in Ybins])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lineprop = {'lw':  kwargs.get('lw', 1)}
    dataprop = {'ms':  kwargs.get('ms', 6),
                'mew': kwargs.get('mew', 0)}

    cohs = sorted(correct_by_coh)
    xall = []
    for coh in cohs:
        stim, correct = correct_by_coh[coh]

        plot.plot(stim, correct, color=colors[coh], label='{}\%'.format(SCALE*coh),
                  **lineprop)
        plot.plot(stim, correct, 'o', mfc=colors[coh], **dataprop)
        xall.append(stim)

    plot.lim('x', xall)
    plot.ylim(50, 100)

#=========================================================================================

def chronometric_function(trialsfile, plot, plot_dist=None, **kwargs):
    """
    Chronometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric performance by RT
    #-------------------------------------------------------------------------------------

    correct_rt_by_coh = {}
    error_rt_by_coh   = {}
    n_below_threshold = 0
    ntot = 0
    for trial in trials:
        info = trial['info']
        coh  = info['coh']

        # Choice and RT
        choice = get_choice(trial, True)
        if choice is None:
            n_below_threshold += 1
            continue
        choice, t_choice = choice
        rt = trial['t'][t_choice] - info['epochs']['stimulus'][0]

        if choice == info['choice']:
            correct_rt_by_coh.setdefault(coh, []).append(rt)
        else:
            error_rt_by_coh.setdefault(coh, []).append(rt)
    print("[ {}.chronometric_function ] {}/{} trials did not reach threshold."
          .format(THIS, n_below_threshold, ntrials))

    # Correct trials
    cohs = np.sort(np.array(correct_rt_by_coh.keys()))
    cohs = cohs[np.where(cohs > 0)]
    correct_rt = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        if len(correct_rt_by_coh[coh]) > 0:
            correct_rt[i] = np.mean(correct_rt_by_coh[coh])
    scaled_cohs = SCALE*cohs

    # Error trials
    cohs = np.sort(np.array(error_rt_by_coh.keys()))
    cohs = cohs[np.where(cohs > 0)]
    error_rt = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        if len(error_rt_by_coh[coh]) > 0:
            error_rt[i] = np.mean(error_rt_by_coh[coh])
    error_scaled_cohs = SCALE*cohs

    #-------------------------------------------------------------------------------------
    # Plot RTs
    #-------------------------------------------------------------------------------------

    # Correct
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    plot.plot(scaled_cohs, correct_rt, **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6),
            'mfc':       '0.2',
            'mew':       0}
    plot.plot(scaled_cohs, correct_rt, **prop)

    # Error
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    #plot.plot(error_scaled_cohs, error_rt, **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6) - 1,
            'mfc':       'w',
            'mec':       '0.2',
            'mew':       kwargs.get('mew', 1)}
    #plot.plot(error_scaled_cohs, error_rt, **prop)

    plot.xscale('log')
    plot.xticks([1, 10, 100])
    plot.xticklabels([1, 10, 100])

    #plot.ylim(300, 1000)

    #-------------------------------------------------------------------------------------
    # Plot RT distribution
    #-------------------------------------------------------------------------------------

    correct_rt = []
    for rt in correct_rt_by_coh.values():
        correct_rt += rt

    error_rt = []
    for rt in error_rt_by_coh.values():
        error_rt += rt

    pdf = plot_dist.hist(correct_rt, color='0.2')
    plot_dist.lim('y', pdf, lower=0)

    print("[ {}.chronometric_function ]".format(THIS))
    print("  Mean RT, correct trials: {:.2f} ms".format(np.mean(correct_rt)))
    print("  Mean RT, error trials:   {:.2f} ms".format(np.mean(error_rt)))

#=========================================================================================

def sort_trials_stim_onset(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get unique conditions
    conds = []
    cohs  = []
    for trial in trials:
        info = trial['info']
        conds.append((info['coh'], info['in_out']))
        cohs.append(info['coh'])
    conds = list(set(conds))
    cohs  = list(set(cohs))

    #-------------------------------------------------------------------------------------
    # Prepare for averaging
    #-------------------------------------------------------------------------------------

    # Number of units
    nunits = trials[0]['r'].shape[0]

    # Number of time points
    stimulus = [np.ptp(trial['info']['epochs']['stimulus']) for trial in trials]
    idx      = np.argmax(stimulus)
    trial    = trials[idx]
    t        = trial['t']
    w        = np.where(t <= trial['info']['epochs']['stimulus'][1])[0][-1] + 1
    t        = t[:w] - trial['info']['epochs']['stimulus'][0]
    ntime    = len(t)

    #-------------------------------------------------------------------------------------
    # Average across conditions
    #-------------------------------------------------------------------------------------

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: np.zeros(ntime)           for c in conds}

    for trial in trials:
        info = trial['info']

        # Include only correct trials
        coh    = info['coh']
        choice = get_choice(trial)

        # Include only correct trials
        if choice != info['choice']:
            continue

        t_i = trial['t']
        w_i = np.where(t_i <= info['epochs']['stimulus'][1])[0][-1] + 1

        c = (info['coh'], info['in_out'])
        sorted_trials[c][:,:w_i] += trial['r'][:,:w_i]
        ntrials_by_cond[c][:w_i]  += 1
    for c in conds:
        sorted_trials[c] *= np.array([safe_divide(x) for x in ntrials_by_cond[c]])

    # Save
    with open(sortedfile, 'wb') as f:
        pickle.dump((t, sorted_trials), f, pickle.HIGHEST_PROTOCOL)
        print(("[ {}.sort_trials_stim_onset ]"
               " Trials sorted and aligned to stimulus onset, saved to {}")
              .format(THIS, sortedfile))

#=========================================================================================

def sort_trials_response(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get unique conditions
    conds = []
    cohs  = []
    for trial in trials:
        info = trial['info']
        conds.append((info['coh'], info['in_out']))
        cohs.append(info['coh'])
    conds = list(set(conds))
    cohs  = list(set(cohs))

    #-------------------------------------------------------------------------------------
    # Determine reaction time on each trial
    #-------------------------------------------------------------------------------------

    valid_trial_idx   = []
    valid_rt_idx      = []
    n_below_threshold = 0
    for i, trial in enumerate(trials):
        info = trial['info']

        # Choice & RT
        choice = get_choice(trial, True)
        if choice is None:
            n_below_threshold += 1
            continue
        choice, w0 = choice

        # Include only correct trials
        if choice != info['choice']:
            continue

        valid_trial_idx.append(i)
        valid_rt_idx.append(w0+1)
    print("[ {}.sort_trials_response ] {}/{} trials did not reach threshold."
          .format(THIS, n_below_threshold, ntrials))

    #-------------------------------------------------------------------------------------
    # Prepare for averaging
    #-------------------------------------------------------------------------------------

    # Number of units
    nunits = trials[0]['r'].shape[0]

    # Number of time points
    idx    = np.argmax(valid_rt_idx)
    w      = valid_rt_idx[idx]
    t      = trials[valid_trial_idx[idx]]['t'][:w]
    t     -= t[-1]
    ntime  = len(t)

    #-------------------------------------------------------------------------------------
    # Average across conditions
    #-------------------------------------------------------------------------------------

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: np.zeros(ntime) for c in conds}
    for i, w in zip(valid_trial_idx, valid_rt_idx):
        trial = trials[i]
        info  = trial['info']

        c = (info['coh'], info['in_out'])
        sorted_trials[c][:,-w:] += trial['r'][:,:w]
        ntrials_by_cond[c][-w:] += 1
    for c in conds:
        sorted_trials[c] *= np.array([safe_divide(x) for x in ntrials_by_cond[c]])

    # Save
    with open(sortedfile, 'wb') as f:
        pickle.dump((t, sorted_trials), f, pickle.HIGHEST_PROTOCOL)
        print(("[ {}.sort_trials_response ]"
               " Trials sorted and aligned to response, saved to {}")
              .format(THIS, sortedfile))

#=========================================================================================

def plot_unit(unit, sortedfile, plot, t0=0, tmin=-np.inf, tmax=np.inf, **kwargs):
    # Load sorted trials
    with open(sortedfile) as f:
        t, sorted_trials = pickle.load(f)

    # Time
    w, = np.where((tmin <= t) & (t <= tmax))
    t  = t[w] - t0

    conds = sorted_trials.keys()

    all = []
    for c in sorted(conds, key=lambda c: c[0]):
        coh, in_out = c

        prop = {'color': colors[coh],
                'lw':    kwargs.get('lw', 1.5)}

        if in_out == +1:
            prop['label'] = '{:.1f}\%'.format(SCALE*coh)
        else:
            prop['linestyle'] = '--'
            prop['dashes'] = kwargs.get('dashes', [3.5, 2.5])

        r = sorted_trials[c][unit][w]
        plot.plot(t, r, **prop)
        all.append(r)

    plot.xlim(t[0], t[-1])
    plot.xticks([t[0], 0, t[-1]])
    plot.lim('y', all, lower=0)

    return np.concatenate(all)

#=========================================================================================

def get_choice_selectivity(trialsfile):
    """
    Compute d' for choice.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

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
        threshold = False
        if 'threshold' in args:
            threshold = True

        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------------------------

        trialsfile = get_trialsfile(p)
        psychometric_function(trialsfile, plot, threshold=threshold)

        plot.xlabel(r'Percent coherence toward $T_\text{in}$')
        plot.ylabel(r'Percent $T_\text{in}$')

        #---------------------------------------------------------------------------------

        fig.save(path=p['figspath'], name=p['name']+'_'+action)
        fig.close()

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    elif action == 'sort_stim_onset':
        sort_trials_stim_onset(get_trialsfile(p), get_sortedfile_stim_onset(p))

    elif action == 'sort_response':
        sort_trials_response(get_trialsfile(p), get_sortedfile_response(p))

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to stimulus onset
    #-------------------------------------------------------------------------------------

    elif action == 'units_stim_onset':
        from glob import glob

        # Remove existing files
        filenames = glob(join(p['figspath'], p['name'] + '_stim_onset_unit*'))
        for filename in filenames:
            os.remove(filename)
            print("Removed {}".format(filename))

        # Load sorted trials
        sortedfile = get_sortedfile_stim_onset(p)
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

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'],
                     name=p['name']+'_stim_onset_unit{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to response
    #-------------------------------------------------------------------------------------

    elif action == 'units_response':
        from glob import glob

        # Remove existing files
        filenames = glob(join(p['figspath'], p['name'] + '_response_unit*'))
        for filename in filenames:
            os.remove(filename)
            print("Removed {}".format(filename))

        # Load sorted trials
        sortedfile = get_sortedfile_response(p)
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

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'],
                     name=p['name']+'_response_unit_{:03d}'.format(i))
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
        print("[ {}.do ] Unrecognized action.".format(THIS))

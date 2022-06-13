"""
This script runs test case 1 of our paper.
We test the performance of the (control) model on White stimuli masked with narrowband noise
used in Betz et al. (2015a). We quantify the edge maps via a ground truth template.

Last update on 04.06.2022
@author: lynnschmittwilken
"""

import numpy as np
import pickle
import os
import time

import parameters as params
from functions import create_stimuli_dict, create_white_gt, octave_intervals, create_dog, \
     create_tfilt, create_drift, run_active_model, run_canny, quantify_edges, remove_borders, \
     plots_exp1

start = time.time()


#########################################
#              Parameters               #
#########################################
plotting = True

# Experimental conditions for model and controls:
# Testing individual model components:
ST_M_N = 'ST_M_N'
S_M_N = 'S_M_N'
T_M = 'T_M'
ST_M = 'ST_M'
S_M = 'S_M'

# Exchanging temporal filtering by variance
S_V_N = 'S_V_N'
S_V = 'S_V'
V = 'V'

# Canny:
canny = 'canny'

# Visual extent of the stimulus
visual_extent = params.visual_extent

# Spatial resolution for simulations. High sf filters might not be depicted well for small ppds.
ppd = params.ppd

# Number of trials:
n_trials = params.n_trials

# Drift model
# Diffusion coefficient in arcmin**2/s (controls drift lengths):
D = params.D
# Total time in s
T = params.T
# Temporal resolution
pps = params.pps
# Temporal resolution for simplified model:
pps_low = params.pps_low

# Stimulus:
# Grating frequencies in White stimulus:
low = 'low'
med = 'medium'
high = 'high'
# Stimulus size:
stimulus_size = int((visual_extent[1] - visual_extent[0]) * ppd)

# Noisemasks:
n_masks = params.n_masks
noisefreqs = params.noisefreqs

# DOG-filters:
# Number of filters used:
n_filters = params.n_filters
# Define the frequency range of filter bank (std in deg):
smax = params.smax

# Create a dict with conditions to be run:
config_dicts = [
    # Testing individual model components:
    {'condition': ST_M_N, 'white_freq': low},
    {'condition': ST_M_N, 'white_freq': med},
    {'condition': ST_M_N, 'white_freq': high},
    {'condition': S_M_N, 'white_freq': low},
    {'condition': S_M_N, 'white_freq': med},
    {'condition': S_M_N, 'white_freq': high},
    {'condition': ST_M, 'white_freq': low},
    {'condition': ST_M, 'white_freq': med},
    {'condition': ST_M, 'white_freq': high},
    {'condition': T_M, 'white_freq': low},
    {'condition': T_M, 'white_freq': med},
    {'condition': T_M, 'white_freq': high},
    # Exchanging temporal filtering by variance
    {'condition': S_V_N, 'white_freq': low},
    {'condition': S_V_N, 'white_freq': med},
    {'condition': S_V_N, 'white_freq': high},
    {'condition': S_V, 'white_freq': low},
    {'condition': S_V, 'white_freq': med},
    {'condition': S_V, 'white_freq': high},
    {'condition': V, 'white_freq': low},
    {'condition': V, 'white_freq': med},
    {'condition': V, 'white_freq': high},
    # Canny
    {'condition': canny, 'white_freq': low},
    {'condition': canny, 'white_freq': med},
    {'condition': canny, 'white_freq': high}
    ]


#########################################
#             Preparation               #
#########################################
# Create outputs folder:
results_path = params.results_path1
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Create stimuli dict with n_trials instantiations to use the same stimuli in all conditions
print('----------------------------------------------')
print('Create stimuli dictionaries ...')

stimuli_high = create_stimuli_dict('high', params)
stimuli_medium = create_stimuli_dict('medium', params)
stimuli_low = create_stimuli_dict('low', params)

print('Done')
print('----------------------------------------------')

# Spatial filters:
# Define spatial frequency axis
nX = stimulus_size
fs = np.fft.fftshift(np.fft.fftfreq(nX, d=1./ppd))
fs_ext = (fs[0], fs[-1], fs[0], fs[-1])

# Define filter freqs in octave intervals:
sigmas = octave_intervals(n_filters) * smax

# Create a list with all dog filters:
dogs = []
for i in range(n_filters):
    # Create DoG filters in the frequency domain that have the same size as the stimulus.
    # Add a third dimension for filtering the 3d input video:
    fx, fy = np.meshgrid(fs, fs)
    dog = create_dog(fx, fy, sigmas[i], 2.*sigmas[i])
    dogs.append(np.expand_dims(dog, -1))

# Define temporal frequency axis:
nT = int(T*pps + 1)
ft = np.fft.fftshift(np.fft.fftfreq(nT, d=1./pps))

# Temporal filter:
tfilt = create_tfilt(ft)
# Add dims for performing 3d fft in space (x, y) and time (t):
tfilt = np.expand_dims(tfilt, (0, 1))


#########################################
#                 Main                  #
#########################################
# Run experiment 1 for each config:
for i in range(len(config_dicts)):
    # This config:
    config = config_dicts[i]

    # Select the right stimuli:
    if config['white_freq'] == 'high':
        stimuli_dict = stimuli_high
    elif config['white_freq'] == 'medium':
        stimuli_dict = stimuli_medium
    elif config['white_freq'] == 'low':
        stimuli_dict = stimuli_low

    # Parameters:
    condition = config['condition']
    white_freq = config['white_freq']
    stimuli = stimuli_dict['stimuli']
    back_size = stimuli_dict['background_size']

    # Prefix for result file names (pickles and figures)
    fprefix = condition + '_' + str(smax) + '_'

    # Create ground truth template:
    edge_thickness = params.edge_thickness * ppd
    gt_template = create_white_gt(stimulus_size, back_size, white_freq, ppd, edge_thickness)

    # Remove borders:
    # We remove borders to avoid border effects that result from how we implement ocular drift
    rb = int(params.remove_b * ppd)
    gt_template = remove_borders(gt_template, rb)

    # Inititate variable for results:
    corrs_trials = np.zeros([n_trials, n_masks])

    # Run simulations:
    for t in range(n_trials):
        print('Trial', t+1)
        stims = stimuli[t]

        # Create drift trajectories in px:
        _, drift = create_drift(T, pps, ppd, D)
        _, drift_low = create_drift(T, pps_low, ppd, D)

        # Compute model outputs:
        if condition == 'ST_M_N':
            # Spatiotemporal filtering followed by squared mean and global normalization
            out = run_active_model(stims, drift, sfilts=dogs, tfilt=tfilt, rb=rb, integrate='mean2', norm=True)

        elif condition == 'S_M_N':
            # Spatial filtering followed by squared mean and global normalization
            out = run_active_model(stims, drift, sfilts=dogs, tfilt=1., rb=rb, integrate='mean2', norm=True)

        elif condition == 'ST_M':
            # Spatiotemporal filtering followed by squared mean and no normalization
            out = run_active_model(stims, drift, sfilts=dogs, tfilt=tfilt, rb=rb, integrate='mean2', norm=False)

        elif condition == 'T_M':
            # Temporal filtering followed by squared mean and no normalization
            out = run_active_model(stims, drift, sfilts=[1.], tfilt=tfilt, rb=rb, integrate='mean2', norm=False)

        elif condition == 'S_V_N':
            # Spatial filtering followed by variance and global normalization
            out = run_active_model(stims, drift_low, sfilts=dogs, tfilt=1., rb=rb, integrate='var', norm=True)

        elif condition == 'S_V':
            # Spatial filtering followed by variance and no normalization
            out = run_active_model(stims, drift_low, sfilts=dogs, tfilt=1., rb=rb, integrate='var', norm=False)

        elif condition == 'V':
            # No filtering followed by variance
            out = run_active_model(stims, drift_low, sfilts=[1.], tfilt=1., rb=rb, integrate='var', norm=False)

        elif condition == 'canny':
            # Canny with optimal parameters:
            out = run_canny(stims, 0., 79.1379, 3)
            out = remove_borders(out, rb)

        # Normalize outputs between 0 and 1 (optional)
        out = out / np.expand_dims(out.max(axis=(0, 1)), (0, 1))

        # Quantify edges of the final model outputs
        corrs = quantify_edges(out, gt_template)
        corrs_trials[t, :] = corrs

    # Create a dict with all relevant parameters and results.
    # For visualization purposes, we save the input stimuli and model outputs for the last trial.
    save_dict = {'visual_extent': visual_extent,
                 'ppd': ppd,
                 'D': D,
                 'T': T,
                 'pps': pps,
                 'pps_low': pps_low,
                 'white_freq': white_freq,
                 'sigmas': sigmas,
                 'noisefreqs': noisefreqs,
                 'edge_thickness': edge_thickness,
                 'remove_b': rb / ppd,
                 'gt_template': gt_template,
                 'n_trials': n_trials,
                 'corrs_trials': corrs_trials,
                 'stimuli': stimuli[n_trials-1],
                 'model_outputs': out}

    # Prepare a name for the pickle file in which we save our results
    pickle_name = fprefix + 'D' + str(np.round(D*60.**2., 1)) + '_T' \
        + str(T) + '_ppd' + str(ppd) + '_' + white_freq + '.pickle'

    # Prepare a name for the plots in which we visualize our results
    plot_name = fprefix + 'D' + str(np.round(D*60.**2., 1)) + '_T' \
        + str(T) + '_ppd' + str(ppd) + '_' + white_freq

    # Save results in pickle:
    with open(results_path + pickle_name, 'wb') as handle:
        pickle.dump(save_dict, handle)

    # Save visualizations of results:
    if plotting:
        plots_exp1(save_dict, results_path + plot_name)

    print('Config ' + str(i+1) + '/' + str(len(config_dicts)) + ' finished ...')
    print('----------------------------------------------')


stop = time.time()
print('Elapsed time: %.2f minutes' % ((stop-start) / 60.))

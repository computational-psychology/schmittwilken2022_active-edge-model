"""
This script runs test case 2 of our paper.
We test the performance of the (control) models on the contour database by
Grigorescu et al. (2003). The database contains 40 grey-level images of size
512x512 pixels, in PGM format. Each image represents a natural image scene and is
accompanied by an associated ground truth contour map drawn by a human.
We quantify the model performance by correlating our model outputs with the
ground truth contour maps that were drawn by humans.

Last update on 04.06.2022
@author: lynnschmittwilken
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import glob
import cv2
import pickle

import parameters as params
from functions import octave_intervals, create_dog, create_tfilt, create_drift, run_active_model, \
     quantify_edges, thicken_edges, remove_borders


start = time.time()

#########################################
#              Parameters               #
#########################################
# Spatial resolution for simulations. High sf filters might not be depicted well for small ppds.
ppd = params.ppd

# Number of trials:
n_trials = params.n_trials

# Sigma of Gaussian noise added to images:
s_noise = params.s_noise

# Drift model
# Ddiffusion coefficient in deg**2/s (controls drift lengths):
D = params.D
# Total time in s
T = params.T
# Temporal resolution
pps = params.pps
pps_low = params.pps_low

# DOG-filters:
# Number of filters used:
n_filters = params.n_filters
# Define the frequency range of filter bank (std in deg):
smax = params.smax

# Edge thickness for gt template and Canny edges
edge_thickness = params.edge_thickness * ppd

# Amount of borders to be removed in the end
rb = int(params.remove_b * ppd)

# Specify folder with input database:
input_path = params.data_path2

# Create outputs folder for results:
results_path = params.results_path2
if not os.path.exists(results_path):
    os.mkdir(results_path)

results_path = results_path + 'noise_' + str(s_noise) + '/'
if not os.path.exists(results_path):
    os.mkdir(results_path)


#########################################
#             Preparation               #
#########################################
# Create a list with all image names of database
img_names = glob.glob(input_path + '*.pgm')
img_names = [f[0:-4] for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

# Load one exemplary input image to get stimulus size:
stimulus = cv2.imread(input_path + img_names[0] + '.pgm')

# Calculate image size in visual degree for the simulations
stimulus_size = stimulus.shape[0]
im_size_h = stimulus.shape[0] / ppd
im_size_w = stimulus.shape[1] / ppd

# Visual extent for the input stimuli:
visual_extent = [-im_size_h/2, im_size_h/2, -im_size_w/2, im_size_w/2]  # in deg

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
tff = np.fft.fftshift(np.fft.fftfreq(nT, d=1./pps))

# Temporal filter:
tfilt = create_tfilt(tff)
# For performing 3d fft in space (x, y) and time (t):
tfilt = np.abs(np.expand_dims(tfilt, (0, 1)))

# Initiate array for edge quantifications
# Testing individual model components:
corrs_ST_M_N = np.zeros([len(img_names), n_trials])
corrs_S_M_N = np.zeros([len(img_names), n_trials])
corrs_ST_M = np.zeros([len(img_names), n_trials])
corrs_T_M = np.zeros([len(img_names), n_trials])

# Exchanging temporal filtering by variance
corrs_S_V_N = np.zeros([len(img_names), n_trials])
corrs_S_V = np.zeros([len(img_names), n_trials])
corrs_V = np.zeros([len(img_names), n_trials])

# Canny
corrs_canny = np.zeros([len(img_names), n_trials])


#########################################
#                 Main                  #
#########################################
# Run simulations for each image in the database:
for img_id in range(len(img_names)):
    img_name = img_names[img_id]

    print('------------------------------------------------')
    print('Stimulus: ' + img_name + ' (' + str(img_id+1) + '/' + str(len(img_names)) + ')')

    # Import stimulus
    stimulus_int = cv2.imread(input_path + img_name + '.pgm')
    stimulus_int = cv2.cvtColor(stimulus_int, cv2.COLOR_BGR2GRAY)

    # Normalize values from 0-255 to 0-1
    stimulus_float = stimulus_int / 255.

    # Import ground truth
    gt_template = cv2.imread(input_path + 'gt/' + img_name + '_gt_binary.pgm')
    gt_template = cv2.cvtColor(gt_template, cv2.COLOR_BGR2GRAY)

    # Normalize values between 0 (no contour) to 1 (contour)
    gt_template = gt_template / 255.
    gt_template = np.abs(gt_template-1.)

    # Thicken the edge signals to make them more comparable with our output
    gt_template = thicken_edges(gt_template, edge_thickness)
    gt_template = remove_borders(gt_template, rb)

    # Calculate image size in visual degree for the simulations
    im_size_h = stimulus.shape[0] / ppd
    im_size_w = stimulus.shape[1] / ppd

    # Visual extent for the input stimuli:
    visual_extent = [-im_size_h/2, im_size_h/2, -im_size_w/2, im_size_w/2]  # in deg

    # Run simulations for n_trials:
    for trial in range(n_trials):
        noise = np.random.normal(0., s_noise, [nX, nX])

        # Add noise and crop values larger than 1 and smaller than 0
        stimulus_noise = stimulus_float + noise
        stimulus_noise[stimulus_noise <= 0.] = 0.
        stimulus_noise[stimulus_noise >= 1.] = 1.

        # Model functions require the input stimuli in list:
        stims = [stimulus_noise]

        # Canny requires input image to be int
        stimuli_int = (stimulus_noise*255.).astype(np.uint8)

        # Create drift trajectories in px:
        _, drift = create_drift(T, pps, ppd, D)
        _, drift_low = create_drift(T, pps_low, ppd, D)

        # Compute model outputs:
        # Spatiotemporal filtering followed by squared mean and global normalization
        out_ST_M_N = run_active_model(stims, drift, sfilts=dogs, tfilt=tfilt, rb=rb, integrate='mean2', norm=True)

        # Spatial filtering followed by squared mean and global normalization
        out_S_M_N = run_active_model(stims, drift_low, sfilts=dogs, tfilt=1., rb=rb, integrate='mean2', norm=True)

        # Spatiotemporal filtering followed by squared mean and no normalization
        out_ST_M = run_active_model(stims, drift, sfilts=dogs, tfilt=tfilt, rb=rb, integrate='mean2', norm=False)

        # Temporal filtering followed by squared mean and no normalization
        out_T_M = run_active_model(stims, drift, sfilts=[1.], tfilt=tfilt, rb=rb, integrate='mean2', norm=False)

        # Spatial filtering followed by variance and global normalization
        out_S_V_N = run_active_model(stims, drift_low, sfilts=dogs, tfilt=1., rb=rb, integrate='var', norm=True)

        # Spatial filtering followed by variance and no normalization
        out_S_V = run_active_model(stims, drift_low, sfilts=dogs, tfilt=1., rb=rb, integrate='var', norm=False)

        # No filtering followed by variance
        out_V = run_active_model(stims, drift, sfilts=[1.], tfilt=1., rb=rb, integrate='var', norm=False)

        # Compute Canny edges with optimal parameters, increase edge thickness & remove borders
        stimuli_int = cv2.GaussianBlur(stimuli_int, (11, 11), 0)
        out_canny = cv2.Canny(stimuli_int, 52.75862, 105.51724)
        out_canny = thicken_edges(out_canny, edge_thickness)
        out_canny = remove_borders(out_canny, rb)

        # Quantify edges by correlating all outputs
        # Testing individual model components:
        corrs_ST_M_N[img_id, trial] = quantify_edges(out_ST_M_N, gt_template)
        corrs_S_M_N[img_id, trial] = quantify_edges(out_S_M_N, gt_template)
        corrs_ST_M[img_id, trial] = quantify_edges(out_ST_M, gt_template)
        corrs_T_M[img_id, trial] = quantify_edges(out_T_M, gt_template)

        # Exchanging temporal filtering by variance
        corrs_S_V_N[img_id, trial] = quantify_edges(out_S_V_N, gt_template)
        corrs_S_V[img_id, trial] = quantify_edges(out_S_V, gt_template)
        corrs_V[img_id, trial] = quantify_edges(out_V, gt_template)

        # Canny
        corrs_canny[img_id, trial] = quantify_edges(out_canny, gt_template)

    #########################################
    #               Plotting                #
    #########################################
    print('All trials done! Now plotting ...')
    n, m = 1, 10
    plt.figure(figsize=(30, 4))
    plt.subplot(n, m, 1)
    plt.imshow(stims[0], cmap='gray', extent=visual_extent)
    plt.title('Input image')

    plt.subplot(n, m, 2)
    plt.imshow(gt_template, extent=visual_extent, cmap='pink')
    plt.title('Human GT')
    plt.savefig(results_path + img_name + '.png')

    plt.subplot(n, m, 3)
    plt.imshow(np.squeeze(out_ST_M_N), extent=visual_extent, cmap='pink')
    plt.title('ST-M-N*, r = ' + str(np.round(corrs_ST_M_N[img_id, :].mean(), 2)))

    plt.subplot(n, m, 4)
    plt.imshow(np.squeeze(out_S_M_N), extent=visual_extent, cmap='pink')
    plt.title('S-M-N, r = ' + str(np.round(corrs_S_M_N[img_id, :].mean(), 2)))

    plt.subplot(n, m, 5)
    plt.imshow(np.squeeze(out_ST_M), extent=visual_extent, cmap='pink')
    plt.title('ST-M, r = ' + str(np.round(corrs_ST_M[img_id, :].mean(), 2)))

    plt.subplot(n, m, 6)
    plt.imshow(np.squeeze(out_T_M), extent=visual_extent, cmap='pink')
    plt.title('T-M, r = ' + str(np.round(corrs_T_M[img_id, :].mean(), 2)))

    plt.subplot(n, m, 7)
    plt.imshow(np.squeeze(out_S_V_N), extent=visual_extent, cmap='pink')
    plt.title('S-V-N*, r = ' + str(np.round(corrs_S_V_N[img_id, :].mean(), 2)))

    plt.subplot(n, m, 8)
    plt.imshow(np.squeeze(out_S_V), extent=visual_extent, cmap='pink')
    plt.title('S-V, r = ' + str(np.round(corrs_S_V[img_id, :].mean(), 2)))

    plt.subplot(n, m, 9)
    plt.imshow(np.squeeze(out_V), extent=visual_extent, cmap='pink')
    plt.title('V, r = ' + str(np.round(corrs_V[img_id, :].mean(), 2)))

    plt.subplot(n, m, 10)
    plt.imshow(np.squeeze(out_canny), extent=visual_extent, cmap='pink')
    plt.title('Canny, r = ' + str(np.round(corrs_canny[img_id, :].mean(), 2)))

    plt.tight_layout()
    plt.savefig(results_path + img_name + '.png', dpi=150)
    plt.close()


#########################################
#             Save pickles              #
#########################################
save_dict = {'ppd': ppd,
             'D': D,
             'T': T,
             'pps': pps,
             'pps_low': pps_low,
             'n_trials': n_trials,
             'corrs_ST_M_N': corrs_ST_M_N,
             'corrs_S_M_N': corrs_S_M_N,
             'corrs_ST_M': corrs_ST_M,
             'corrs_T_M': corrs_T_M,
             'corrs_S_V_N': corrs_S_V_N,
             'corrs_S_V': corrs_S_V,
             'corrs_V': corrs_V,
             'corrs_canny': corrs_canny,
             'noise': noise}

pickle_file = 'edge_corr.pickle'

with open(results_path + pickle_file, 'wb') as handle:
    pickle.dump(save_dict, handle)


#########################################
#           Print statistics            #
#########################################
# Calculate average over trials
corrs_ST_M_N_m = corrs_ST_M_N.mean(1)
corrs_S_M_N_m = corrs_S_M_N.mean(1)
corrs_ST_M_m = corrs_ST_M.mean(1)
corrs_T_M_m = corrs_T_M.mean(1)

corrs_S_V_N_m = corrs_S_V_N.mean(1)
corrs_S_V_m = corrs_S_V.mean(1)
corrs_V_m = corrs_V.mean(1)

corrs_canny_m = corrs_canny.mean(1)

# Print mean and std over images:
print('------------------------------------------------')
print('Statistics (mean and std) ')
print('ST-M-N:', np.round(corrs_ST_M_N_m.mean(), 2), '+/-', np.round(corrs_ST_M_N_m.std(), 2))
print('S-M-N:', np.round(corrs_S_M_N_m.mean(), 2), '+/-', np.round(corrs_S_M_N_m.std(), 2))
print('ST-M:', np.round(corrs_ST_M_m.mean(), 2), '+/-', np.round(corrs_ST_M_m.std(), 2))
print('T-M:', np.round(corrs_T_M_m.mean(), 2), '+/-', np.round(corrs_T_M_m.std(), 2))
print('------------------------------------------------')
print('S-V-N:', np.round(corrs_S_V_N_m.mean(), 2), '+/-', np.round(corrs_S_V_N_m.std(), 2))
print('S-V:', np.round(corrs_S_V_m.mean(), 2), '+/-', np.round(corrs_S_V_m.std(), 2))
print('V:', np.round(corrs_V_m.mean(), 2), '+/-', np.round(corrs_V_m.std(), 2))
print('------------------------------------------------')
print('Canny & GT:', np.round(corrs_canny_m.mean(), 2), '+/-', np.round(corrs_canny_m.std(), 2))

stop = time.time()
print('------------------------------------------------')
print('Elapsed time: %.2f minutes' % ((stop-start) / 60.))

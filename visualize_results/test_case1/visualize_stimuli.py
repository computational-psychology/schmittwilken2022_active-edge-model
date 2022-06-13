"""
Script to visualize the noisemasking stimuli used in experiment 1

Created on 07.04.2021
Last update on 20.04.2021
@author: lynn schmittwilken
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Let's read in params and functions from experiments:
sys.path.append('../../simulations')
import parameters as params
from functions import add_background, create_white_gt, create_white_stimulus, create_noisemask


# Create two-target White stimulus
def create_two_target_white(n=600, nBars=6):
    # Choose values for grating bars
    grating_vals = [0.95, 1.05]

    barwidth = int(n/nBars)            # Width of grating bars
    max_lum = grating_vals[0]          # max luminance value (arbitrary)
    min_lum = grating_vals[1]          # min luminance value (arbitrary)
    t_lum = (max_lum + min_lum) / 2.   # target luminance

    # Create grating:
    stimulus = np.zeros([n, n])
    for i in range(0, nBars, 2):
        stimulus[i*barwidth:(i+1)*barwidth, :] = min_lum
        stimulus[(i+1)*barwidth:(i+2)*barwidth, :] = max_lum

    # Add gray targets:
    x_t1, y_t1 = int((nBars/2) * barwidth), int(n*0.8 - barwidth)
    x_t2, y_t2 = int((nBars/2 - 1) * barwidth), int(n*0.2)
    stimulus[x_t1:x_t1+barwidth, y_t1:y_t1+barwidth] = t_lum
    stimulus[x_t2:x_t2+barwidth, y_t2:y_t2+barwidth] = t_lum
    return stimulus


#########################################
#              Parameters               #
#########################################
plot_stims = True
plot_gt = False
white_version = 'double'  # options: single or double

# Resolution for simulations. Keep in mind that the high spatial
# frequency filters might not be depicted well for small ppds.
ppd = params.ppd

# White stimulus
# Visual extent for the input stimuli:
visual_extent = params.visual_extent

# In Betz2015, three different stimuli were used that were either 10.2deg with
# 4 bars or 7.66deg with 6 or 12 bars (low sf, medium sf, high sf)
white_freq = 'medium'  # 'low', 'medium' or 'high'

# Noisefrequencies for noisemasks (Parameters taken from Betz2015)
noisefreqs = params.noisefreqs
n_masks = params.n_masks

# Edge thickness for ground truth edge map
edge_thickness = params.edge_thickness * ppd

# Create outputs folder:
result_folder = 'stimuli/'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)


#########################################
#                 Main                  #
#########################################
# For the visualizations, let's increase the contrast of the grating
f_contrast = 3.

# Create white stimulus in accordance to Betz2015
if white_version == 'single':
    white_stimulus = create_white_stimulus(white_freq, ppd) * f_contrast
elif white_version == 'double':
    white_stimulus = create_two_target_white(int(1.276 * ppd * 6)) * f_contrast
white_size = white_stimulus.shape[0]

# Add gray background that covers the desired visual extent
background_size = 2. * ppd
# background_size = visual_extent[1]*2.*ppd-white_size
white_stimulus = add_background(white_stimulus, background_size, back_lum=1.*f_contrast)
stimulus_size = white_stimulus.shape[0]

# Create list with all stimuli:
stimuli = []
for i in range(n_masks):
    # Apply noise mask of given noisefreq
    stimuli.append(white_stimulus + create_noisemask(stimulus_size, noisefreqs[i], ppd))


# Create ground truth template:
gt_template = create_white_gt(stimulus_size, background_size, white_freq, ppd, edge_thickness)


# Plot and save plots
if plot_stims:
    freq = 0
    plt.figure(figsize=(10, 8))
    plt.imshow(stimuli[freq], cmap='gray', extent=visual_extent)
    plt.axis('off')
    plt.savefig(result_folder + white_freq + '_' + str(noisefreqs[freq]) + '.png', dpi=300)
    plt.close()
    
    freq = 3
    plt.figure(figsize=(10, 8))
    plt.imshow(stimuli[freq], cmap='gray', extent=visual_extent)
    plt.axis('off')
    plt.savefig(result_folder + white_freq + '_' + str(noisefreqs[freq]) + '.png', dpi=300)
    plt.close()
    
    freq = n_masks - 1
    plt.figure(figsize=(10, 8))
    plt.imshow(stimuli[freq], cmap='gray', extent=visual_extent)
    plt.axis('off')
    plt.savefig(result_folder + white_freq + '_' + str(noisefreqs[freq]) + '.png', dpi=300)
    plt.close()

if plot_gt:
    plt.figure(figsize=(10, 8))
    plt.imshow(gt_template, cmap='pink', extent=visual_extent)
    plt.axis('off')
    plt.savefig(result_folder + white_freq + '_gt.png', dpi=300)
    plt.close()

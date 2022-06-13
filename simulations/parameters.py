"""
File that contains all parameters for simulations.

Last update on 04.06.2022
@author: lynnschmittwilken
"""

###########################
#         General         #
###########################
# Path for saving results of experiment 1:
results_path1 = 'results_case1/'

# Path for saving results of experiment 2:
results_path2 = 'results_case2/'

# Path for loading Contour Image database in test case 2:
data_path2 = '../databases/grigorescu2003_contour_image_database/images/'

# Spatial resolution for simulations (pixels per degree).
# High spatial frequency filters might not be depicted well for too small ppds.
ppd = 40.

# Number of trials in experiments
n_trials = 10


###########################
#          Drift          #
###########################
# Diffusion coefficient in deg**2/s (controls drift lengths)
D = 20. / (60.**2.)

# Total simulated fixation time in s
T = 0.2

# Drift sampling frequency in Hz / Temporal resolution
pps = 100.

# If we do not perform temporal filtering, we can decrease the temporal sampling frequency
pps_low = 10.


###########################
#          Models         #
###########################
# Number of filters in multiscale filter bank:
n_filters = 5

# Frequency ranges of filter banks (stds in deg):
smax = 0.256

# Edge thickness for ground truth maps (deg):
# It roughly matches the edge thickness in our model outputs
edge_thickness = 0.1

# Amount of borders to be removed (deg) to avoid border effects
remove_b = 0.5


###########################
#      White stimuli      #
###########################
# Visual extent of stimulus in test case 1 (deg)
visual_extent = [-8., 8., -8., 8.]  # in deg

# Noise center frequencies in cpd (Parameters taken from Betz2015)
noisefreqs = [0.58, 1., 1.73, 3., 5.2, 9.]

# Number of noise masks:
n_masks = len(noisefreqs)


###########################
#     Natural stimuli     #
###########################
# Sigma of Gaussian white noise added
s_noise = 0.1  # either 0. or 0.1

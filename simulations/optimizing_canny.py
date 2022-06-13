"""
Script to optimize performance of Canny edge detector

Last update on 08.06.2022
@author: lynnschmittwilken
"""

import numpy as np
import os
import glob
import cv2
import time
import pickle

import parameters as params
from functions import run_canny, add_background, remove_borders, quantify_edges, thicken_edges, \
    create_white_gt, print_progress, create_white_stimulus, create_noisemask


####################################
#           Parameters             #
####################################
# Find optimal parameters for case1 or case2?
test = 'case2'
pickle_file = 'optimal_canny_' + test + '.pickle'

# Spatial resolution for simulations (pixels per degree)
ppd = params.ppd
edge_thickness = params.edge_thickness * ppd
remove_b = int(params.remove_b * params.ppd)


####################################
#            Functions             #
####################################
def test_case1_stimuli():
    """Helper function that creates all stimuli for test case 1 that we will use
    for optimizing the Canny edge detector

    Returns
    -------
    stimuli
        List with stimuli
    gt_template
        Ground truth template

    """
    # Visual extent of full stimulus (deg)
    visual_extent = params.visual_extent

    # Noisefrequencies of masks in cpd (Parameters taken from Betz2015)
    noisefreqs = params.noisefreqs
    n_masks = len(noisefreqs)

    # Chosen grating frequency
    white_freq = 'medium'

    # Create white stimulus in accordance to Betz2015
    white_stimulus = create_white_stimulus(white_freq, ppd)
    white_size = white_stimulus.shape[0]

    # Add gray background that covers the desired visual extent
    back_size = visual_extent[1]*2.*ppd-white_size
    white_stimulus = add_background(white_stimulus, back_size, back_lum=1.)
    stimulus_size = white_stimulus.shape[0]

    # Create ground truth template:
    gt_template = create_white_gt(stimulus_size, back_size, white_freq, ppd, edge_thickness)

    # Remove borders:
    gt_template = remove_borders(gt_template, remove_b)

    # Create list with all stimuli
    stimuli = []
    for j in range(n_masks):
        # Apply noise mask of given noisefreq
        stimuli.append(white_stimulus + create_noisemask(stimulus_size, noisefreqs[j], ppd))
    return stimuli, gt_template


def test_case1(stimuli, gt_template, p):
    """Helper function that runs a Canny edge detector with given parameters on
    a list of stimuli and calculates average edge detection performance as average
    correlation between model outputs and a ground truth template

    Parameters
    ----------
    stimuli
        List of stimuli
    gt_template
        Ground truth template
    p
        List of parameters for current iteration

    Returns
    -------
    performance
        Average performance of Canny edge detector on input stimuli

    """
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    # Run Canny
    model_outputs = run_canny(stimuli, p1, p2, p3, False)

    # Increase edge thickness
    model_outputs = thicken_edges(model_outputs, edge_thickness)

    # Remove borders and normalize between 0 and 1
    model_outputs = remove_borders(model_outputs, remove_b)
    model_outputs = model_outputs / np.expand_dims(model_outputs.max(axis=(0, 1)), (0, 1))

    # Quantify edges of the final model outputs
    corrs = quantify_edges(model_outputs, gt_template)
    performance = corrs.mean()
    return performance


def load_images(n_stims: int):
    """Helper function that loads and prepares n_stims images from the Contour Image
    Database (Grigorescu et al., 2003) and their respective human-drawn ground truth maps

    Parameters
    ----------
    n_stims
        Number of images to read from the database

    Returns
    -------
    images
        List with n_stims images from the database
    gts
        List with n_stims ground truth maps from the database

    """
    # Specify folder with input database:
    inpath = params.data_path2

    # Create a list with all image names of database
    imnames = glob.glob(inpath + '*.pgm')
    imnames = [f[0:-4] for f in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, f))]

    images = []
    gts = []
    for img_id in range(n_stims):
        img_name = imnames[img_id]

        # Import stimuli
        stimulus_int = cv2.imread(inpath + img_name + '.pgm')
        stimulus_int = cv2.cvtColor(stimulus_int, cv2.COLOR_BGR2GRAY)
        images.append(stimulus_int)

        # Import ground truths
        gt_template = cv2.imread(inpath + 'gt/' + img_name + '_gt_binary.pgm')
        gt_template = cv2.cvtColor(gt_template, cv2.COLOR_BGR2GRAY)

        # Normalize values between 0 (no contour) to 1 (contour)
        gt_template = gt_template / 255.
        gt_template = np.abs(gt_template-1.)

        # Thicken the edge signals to make them more comparable with our output
        gt_template = thicken_edges(gt_template, edge_thickness)
        gt_template = remove_borders(gt_template, remove_b)
        gts.append(gt_template)
    return images, gts


def test_case2(images, gts, p):
    """Helper function that runs a Canny edge detector with given parameters on
    a list of stimuli with and without additional Gaussian white noise and
    calculates average edge detection performance as average correlation
    between model outputs and a ground truth template

    Parameters
    ----------
    images
        List of images from the Contour Image Database
    gts
        List of corresponding ground truth maps
    p
        List of parameters for current iteration

    Returns
    -------
    performance
        Average performance of Canny edge detector on input stimuli with and without
        additional Gaussian white noise

    """
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    # Optimize for noise and no-noise condition. Sigma of Gaussian white noise:
    s_noise = 0.1

    corrs1 = 0
    corrs2 = 0
    n_stims = len(images)

    # Run Canny for all images (by default: 50% of database):
    for img_id in range(n_stims):
        stimulus_int = images[img_id]
        gt_template = gts[img_id]
        nX = stimulus_int.shape[0]

        # Normalize values from 0-255 to 0-1
        stimulus_float = stimulus_int / 255.

        # Add noise and crop values outside desired range
        noise = np.random.normal(0., s_noise, [nX, nX])
        stimulus_noise = stimulus_float + noise
        stimulus_noise[stimulus_noise <= 0.] = 0.
        stimulus_noise[stimulus_noise >= 1.] = 1.

        # Prepare stimuli with / without noise
        stimuli_int1 = stimulus_int
        stimuli_int2 = (stimulus_noise*255.).astype(np.uint8)

        # Compute Canny edges & increase edge thickness
        stimuli_int1 = cv2.GaussianBlur(stimuli_int1, (p3, p3), 0)
        out_canny1 = cv2.Canny(stimuli_int1, p1, p2)
        out_canny1 = thicken_edges(out_canny1, edge_thickness)

        stimuli_int2 = cv2.GaussianBlur(stimuli_int2, (p3, p3), 0)
        out_canny2 = cv2.Canny(stimuli_int2, p1, p2)
        out_canny2 = thicken_edges(out_canny2, edge_thickness)

        # Remove borders and normalize between 0 and 1
        out_canny1 = remove_borders(out_canny1, remove_b)
        out_canny1 = out_canny1 / np.expand_dims(out_canny1.max(axis=(0, 1)), (0, 1))

        out_canny2 = remove_borders(out_canny2, remove_b)
        out_canny2 = out_canny2 / np.expand_dims(out_canny2.max(axis=(0, 1)), (0, 1))

        # Quantify edges of the final model outputs
        corrs1 += quantify_edges(out_canny1, gt_template)
        corrs2 += quantify_edges(out_canny2, gt_template)

    performance = (corrs1+corrs2) / n_stims / 2.
    return performance


####################################
#           Simulations            #
####################################
start = time.time()

# Check whether pickle exists. If so, load data from pickle file and continue optimization
if os.path.isfile(pickle_file):
    print('Loading existing optimization data from pickle...')
    print()
    # Load data from pickle:
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    p1s = data['p1s']
    p2s = data['p2s']
    p3s = data['p3s']
    best_performance = data['best_performance']
    best_params = data['best_params']
    last_set = data['last_set']

else:
    print('Initiating optimization from scratch...')
    print()
    p1s = np.linspace(0., 255., 30)
    p2s = np.linspace(0., 255., 30)
    p3s = np.arange(3, 21, 2)
    best_performance = 0.
    last_set = 0


# Prepare list of params to run:
total = len(p1s)*len(p2s)*len(p3s)
count = 0
ps = np.zeros([total, 3])
for i in range(len(p1s)):
    for j in range(len(p2s)):
        for k in range(len(p3s)):
            ps[count, 0] = p1s[i]
            ps[count, 1] = p2s[j]
            ps[count, 2] = p3s[k]
            count += 1

# Load images and ground truths for test case 1:
if test == 'case1':
    stimuli_t1, gts_t1 = test_case1_stimuli()

# Load images and ground truths for test case 2:
if test == 'case2':
    stimuli_t2, gts_t2 = load_images(20)

for i in range(last_set, total):
    # Print progress and get relevant params:
    print_progress(count=i+1, total=total)
    p1 = ps[i, 0]
    p2 = ps[i, 1]
    p3 = int(ps[i, 2])
    p = [p1, p2, p3]

    # Low threshold param should be lower than high threshold param:
    if p1 < p2:
        try:
            if test == 'case1':
                performance = test_case1(stimuli_t1, gts_t1, p)
            elif test == 'case2':
                performance = test_case2(stimuli_t2, gts_t2, p)

            if performance > best_performance:
                best_performance = performance
                best_params = [p1, p2, p3]
                print()
                print('New best performance:', best_performance)
                print('Parameters: ', best_params)
                print()
        except:
            # If no edges are found, an error will occur. Ignore and continue.
            pass

        # Save params and results in pickle:
        save_dict = {'p1s': p1s,
                     'p2s': p2s,
                     'p3s': p3s,
                     'last_set': i,
                     'best_performance': best_performance,
                     'best_params': best_params}

        with open(pickle_file, 'wb') as handle:
            pickle.dump(save_dict, handle)

    else:
        pass


stop = time.time()
print()
print('------------------------------------------------')
print('Elapsed time: %.2f minutes' % ((stop-start) / 60.))

"""
These functions are the basis for the implementation of my edge-sensitivity model.
They are also relevant for quantifying the edge quality of the model.

Last update on 04.06.2022
@author: lynnschmittwilken
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import cv2
from scipy import signal

try:
    # Recommended package to speed-up simulations
    import mkl_fft as fft
except:
    import numpy.fft as fft


# %%
###############################
#       Helper functions      #
###############################
def print_progress(count, total):
    """Helper function to print progress.

    Parameters
    ----------
    count
        Current iteration count.
    total
        Total number of iterations.

    """
    percent_complete = float(count) / float(total)
    msg = "\rProgress: {0:.1%}".format(percent_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def octave_intervals(num: int):
    """Helper function to create a 1d array of numbers in octave intervals with a
    a maximum of 1.

    Parameters
    ----------
    num
        Number of elements in octave intervals.

    Returns
    -------
    x
        1d array of numbers in octave intervals

    """
    # Octave intervals means x_i : x_i+1 = 1:2
    x = np.logspace(1, num, num=num, base=2)

    # Normalize, so that maximum is 1
    return x/x.max()


def remove_borders(array, rb: int):
    """Helper function that removes the outer proportion of the first two dimensions
    of the input array. Used to prevent boundary effects.

    Parameters
    ----------
    array
        An array with 2-4 dimensions
    rb
        Number of pixels to be removed at the boundaries

    Returns
    -------
    array
        Array with removed boundaries in the first two dimensions

    """
    array_shape = array.shape
    if len(array_shape) == 2:
        array = array[rb:array_shape[0]-rb, rb:array_shape[1]-rb]
    elif len(array_shape) == 3:
        array = array[rb:array_shape[0]-rb, rb:array_shape[1]-rb, :]
    else:
        array = array[rb:array_shape[0]-rb, rb:array_shape[1]-rb, :, :]
    return array


# Function to increase the thickness of edge signals
def thicken_edges(array, n: int):
    """Helper function that increases the thickness of non-zero elements (here: edges)
    in the input. Used to increase the thickness of edges.

    Parameters
    ----------
    array
        An array with 2-3 dimensions.
    n
        Number of pixels by which the thickness should be increased.

    Returns
    -------
    output
        Array with increased thickness of non-zero elements.

    """
    ashape = array.shape
    add = 50

    if len(ashape) == 2:
        array_large = np.zeros([ashape[0]+add+n, ashape[1]+add+n])

        # Increase width of the contours by shifting the image by n pixels in each direction
        for i in range(int(n/2)):
            array_large[add+i:ashape[0]+add+i, add+i:ashape[1]+add+i] += array
            array_large[add-i:ashape[0]+add-i, add-i:ashape[1]+add-i] += array
            array_large[add+i:ashape[0]+add+i, add:ashape[1]+add] += array
            array_large[add-i:ashape[0]+add-i, add:ashape[1]+add] += array
            array_large[add:ashape[0]+add, add+i:ashape[1]+add+i] += array
            array_large[add:ashape[0]+add, add-i:ashape[1]+add-i] += array

        array_large[array_large > 1.] = 1.
        output = array_large[add:ashape[0]+add, add:ashape[1]+add]

    elif len(ashape) == 3:
        array_large = np.zeros([ashape[0]+add+n, ashape[1]+add+n, ashape[2]])

        # Increase width of the contours by shifting the image by n pixels in each direction
        for i in range(int(n/2)):
            array_large[add+i:ashape[0]+add+i, add+i:ashape[1]+add+i, :] += array
            array_large[add-i:ashape[0]+add-i, add-i:ashape[1]+add-i, :] += array
            array_large[add+i:ashape[0]+add+i, add:ashape[1]+add, :] += array
            array_large[add-i:ashape[0]+add-i, add:ashape[1]+add, :] += array
            array_large[add:ashape[0]+add, add+i:ashape[1]+add+i, :] += array
            array_large[add:ashape[0]+add, add-i:ashape[1]+add-i, :] += array

        array_large[array_large > 1.] = 1.
        output = array_large[add:ashape[0]+add, add:ashape[1]+add, :]
    return output


# %%
###############################
#           Filters           #
###############################
def gauss_fft(fx, fy, sigma: float):
    """Function to create an isotropic Gaussian filter in the frequency domain.

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    sigma
        Sigma that defines the spread of Gaussian filter in deg.

    Returns
    -------
    gauss
        2D Gaussian filter in frequency domain.

    """
    gauss = np.exp(-2. * np.pi**2. * sigma**2. * (fx**2. + fy**2.))
    return gauss


def create_dog(fx, fy, sigma_c, sigma_s):
    """Function to create an isotropic Difference-of-Gaussian filter in the frequency domain

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    sigma_c
        Sigma that defines the spread of the central Gaussian in deg.
    sigma_s
        Sigma that defines the spread of the surround Gaussian in deg.

    Returns
    -------
    dog
        2D Difference-of-Gaussian filter in frequency domain.

    """
    center = gauss_fft(fx, fy, sigma_c)
    surround = gauss_fft(fx, fy, sigma_s)

    dog = center - surround
    return dog


def calculate_peak_freq(dog, fs):
    """Function to calculate peak frequency of a DoG filters defined in frequency space.

    Parameters
    ----------
    dog
        List of 2D Difference-of-Gaussians filters in frequency space.
    fs
        Array with corresponding frequencies.

    Returns
    -------
    peak_freqs
        List of peak frequencies of DoG filters.

    """
    n_filters = len(dog)
    nX = np.size(dog, 0)

    peak_freqs = np.zeros(n_filters)
    for i in range(n_filters):
        filter_row = dog[i][int(nX/2), :]
        max_index = np.where(filter_row == np.max(filter_row))
        max_index = max_index[0][0]
        peak_freqs[i] = np.abs(fs[max_index])
    return peak_freqs


def create_tfilt(tf):
    """Function to create a temporal bandpass filter fitted to the temporal tuning
    properties of macaque simple cells reported in Zheng et al. (2007)

    Parameters
    ----------
    tf
        1d array with temporal frequencies.

    Returns
    -------
    H
        1d temporal bandpass filter.

    """
    # To get a symmetrical filter around 0 Hz, we calculate the absolute tfs:
    tf = np.abs(tf)

    # The equation does not allow tf=0 Hz, so we implement a small workaround and set it to 0
    # manually afterwards
    idx0 = np.where(tf == 0.)[0]
    tf[tf == 0.] = 1.

    # Parameters from fitting the equation to the data of adult macaque V1 cells of Zheng2007:
    m1 = 1.   # 69.3 to actually scale it to the data
    m2 = 22.9
    m3 = 8.1
    m4 = 0.8
    H = m1 * np.exp(-(tf / m2) ** 2.) / (1. + (m3 / tf)**m4)

    if len(idx0):
        H[idx0[0]] = 0.
    return H


def bandpass_filter(fx, fy, fcenter, sigma):
    """Function to create a bandpass filter

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    fcenter
        Center frequency of the bandpass filter
    sigma
        Sigma that defines the spread of the Gaussian in deg.

    Returns
    -------
    dog
        2D Difference-of-Gaussian filter in frequency domain.

    """
    # Calculate the distance of each 2d spatial frequency from requested center frequency
    distance = np.abs(fcenter - np.sqrt(fx**2. + fy**2.))

    # Create bandpass filter:
    bpfilt = 1. / (np.sqrt(2.*np.pi) * sigma) * np.exp(-(distance**2.) / (2.*sigma**2.))
    bpfilt = bpfilt / bpfilt.max()
    return bpfilt


# %%
###############################
#            Drift            #
###############################
def brownian(T: float, pps: float, D: float):
    """Function to create 2d Brownian motion.

    Parameters
    ----------
    T
        Time interval (unit: s)
    pps
        Temporal sampling frequency (unit: Hz)
    D
        Diffusion coefficient (unit: deg**2 / s)

    Returns
    -------
    y
        Displacement array (unit: deg)

    """

    n = int(T*pps)  # Number of drift movements
    dt = 1. / pps   # Time step between two consequent steps (unit: seconds)

    # Generate a 2d stochastic, normally-distributed time series:
    y = np.random.normal(0, 1., [2, n])

    # The average displacement is proportional to dt and D
    y = y * np.sqrt(2.*dt*D)

    # Set initial displacement to 0.
    y = np.insert(y, 0, 0., axis=1)
    return y


def create_drift(T, pps, ppd, D):
    """Function to create 2d drift motion.

    Parameters
    ----------
    T
        Time interval (unit: s)
    pps
        Temporal sampling frequency (unit: Hz)
    ppd
        Spatial resolution (pixels per degree)
    D
        Diffusion coefficient (unit: deg**2 / s)

    Returns
    -------
    y
        Continuous drift array (unit: "continuous px")
    y_int
        Discretized drift array (unit: px)

    """

    # Since our simulations are in px-space, we want to ensure that our drift paths != 0
    cond = 0.
    while (cond == 0.):
        # Generate 2d brownian displacement array
        y = brownian(T, pps, D) * ppd

        # Generate drift path in px from continuous displacement array
        y = np.cumsum(y, axis=-1)
        y_int = np.round(y).astype(int)

        # Sum the horizontal and vertical drift paths in px to make sure that both are != 0:
        cond = y_int[0, :].sum() * y_int[1, :].sum()
    return y, y_int


def apply_drift(stimulus, drift, back_lum=0.5):
    """Create a video in which the stimulus gets shifted over time based on drift.

    Parameters
    ----------
    stimulus
        2D array / image
    drift
        2D discretized drift array (unit: px)
    back_lum
        Intensity value that is used to extend the array

    Returns
    -------
    stimulus_video
        3D array (dimensions: x, y, t) in which the stimulus is shifted over time

    """
    steps = np.size(drift, 1)
    center_x1 = int(np.size(stimulus, 0) / 2)
    center_y1 = int(np.size(stimulus, 1) / 2)

    # Determine the largest displacement and increase stimulus size accordingly
    largest_disp = int(np.abs(drift).max())
    stimulus_extended = np.pad(stimulus, largest_disp, 'constant', constant_values=(back_lum))
    center_x2 = int(np.size(stimulus_extended, 0) / 2)
    center_y2 = int(np.size(stimulus_extended, 1) / 2)

    # Initialize drift video:
    stimulus_video = np.zeros([np.size(stimulus, 0), np.size(stimulus, 1), steps], np.float16)
    stimulus_video[:, :, 0] = stimulus

    for t in range(1, steps):
        x, y = int(drift[0, t]), int(drift[1, t])

        # Create drift video:
        stimulus_video[:, :, t] = stimulus_extended[
                center_x2-center_x1+x:center_x2+center_x1+x,
                center_y2-center_y1+y:center_y2+center_y1+y]
    return stimulus_video.astype(np.float32)


# %%
###############################
#       Stimulus-related      #
###############################
def randomize_sign(array):
    """Helper function that randomizes the sign of values in an array.

    Parameters
    ----------
    array
        N-dimensional array

    Returns
    -------
    array
        Same array with randomized signs

    """
    sign = np.random.rand(*array.shape) - 0.5
    sign[sign <= 0.] = -1.
    sign[sign > 0.] = 1.
    array = array * sign
    return array


def pseudo_white_noise_patch(shape, A):
    """Helper function used to generate pseudorandom white noise patch.

    Parameters
    ----------
    shape
        Shape of noise patch
    A
        Amplitude of each (pos/neg) frequency component = A/2

    Returns
    -------
    output
        Pseudorandom white noise patch

    """
    Re = np.random.rand(*shape) * A - A/2.
    Im = np.sqrt((A/2.)**2 - Re**2)
    Im = randomize_sign(Im)
    output = Re+Im*1j
    return output


def pseudo_white_noise(n, A=2.):
    """Function to create pseudorandom white noise. Code translated and adapted
    from Matlab scripts provided by T. Peromaa

    Parameters
    ----------
    n
        Even-numbered size of output
    A
        Amplitude of noise power spectrum

    Returns
    -------
    spectrum
        Shifted 2d complex number spectrum. DC = 0.
        Amplitude of each (pos/neg) frequency component = A/2
        Power of each (pos/neg) frequency component = (A/2)**2

    """
    # We divide the noise spectrum in four quadrants with pseudorandom white noise
    quadrant1 = pseudo_white_noise_patch((int(n/2)-1, int(n/2)-1), A)
    quadrant2 = pseudo_white_noise_patch((int(n/2)-1, int(n/2)-1), A)
    quadrant3 = quadrant2[::-1, ::-1].conj()
    quadrant4 = quadrant1[::-1, ::-1].conj()

    # We place the quadrants in the spectrum to eventuate that each frequency component has
    # an amplitude of A/2
    spectrum = np.zeros([n, n], dtype=complex)
    spectrum[1:int(n/2), 1:int(n/2)] = quadrant1
    spectrum[1:int(n/2), int(n/2)+1:n] = quadrant2
    spectrum[int(n/2+1):n, 1:int(n/2)] = quadrant3
    spectrum[int(n/2+1):n, int(n/2+1):n] = quadrant4

    # We need to fill the rows / columns that the quadrants do not cover
    # Fill first row:
    row = pseudo_white_noise_patch((1, n), A)
    apu = np.fliplr(row)
    row[0, int(n/2+1):n] = apu[0, int(n/2):n-1].conj()
    spectrum[0, :] = np.squeeze(row)

    # Fill central row:
    row = pseudo_white_noise_patch((1, n), A)
    apu = np.fliplr(row)
    row[0, int(n/2+1):n] = apu[0, int(n/2):n-1].conj()
    spectrum[int(n/2), :] = np.squeeze(row)

    # Fill first column:
    col = pseudo_white_noise_patch((n, 1), A)
    apu = np.flipud(col)
    col[int(n/2+1):n, 0] = apu[int(n/2):n-1, 0].conj()
    spectrum[:, int(n/2)] = np.squeeze(col)

    # Fill central column:
    col = pseudo_white_noise_patch((n, 1), A)
    apu = np.flipud(col)
    col[int(n/2+1):n, 0] = apu[int(n/2):n-1, 0].conj()
    spectrum[:, 0] = np.squeeze(col)

    # Set amplitude at filled-corners to A/2:
    spectrum[0, 0] = -A/2 + 0j
    spectrum[0, int(n/2)] = -A/2 + 0j
    spectrum[int(n/2), 0] = -A/2 + 0j

    # Set DC = 0:
    spectrum[int(n/2), int(n/2)] = 0 + 0j
    return spectrum


def create_noisemask(nX, noisefreq, ppd, rms_contrast=0.2, pseudo_noise=True):
    """Function to create narrowband noise.

    Parameters
    ----------
    nX
        Size of noise image.
    noisefreq
        Noise center frequency in cpd.
    ppd
        Spatial resolution (pixels per degree).
    rms_contrast
        rms contrast of noise.
    pseudo_noise
        Bool, if True generate pseudorandom noise with perfectly smooth
        power spectrum.

    Returns
    -------
    narrow_noise
        2D array with narrowband noise.

    """
    # We calculate sigma either to eventuate a ratio bandwidth of 1 octave
    sigma = noisefreq / (3.*np.sqrt(2.*np.log(2.)))

    # Prepare spatial frequency axes and create bandpass filter:
    fs = np.fft.fftshift(np.fft.fftfreq(nX, d=1./ppd))
    fx, fy = np.meshgrid(fs, fs)
    bp_filter = bandpass_filter(fx, fy, noisefreq, sigma)

    if pseudo_noise:
        # Create white noise with frequency amplitude of 1 everywhere
        white_noise_fft = pseudo_white_noise(nX)
    else:
        # Create white noise and fft
        white_noise = np.random.rand(nX, nX) * 2. - 1.
        white_noise_fft = np.fft.fftshift(np.fft.fft2(white_noise))

    # Filter white noise with bandpass filter
    narrow_noise_fft = white_noise_fft * bp_filter

    # ifft
    narrow_noise = np.fft.ifft2(np.fft.ifftshift(narrow_noise_fft))
    narrow_noise = np.real(narrow_noise)
    narrow_noise = rms_contrast * narrow_noise / narrow_noise.std()
    return narrow_noise


def create_white_stimulus(white_freq: str, ppd):
    """Function to create White stimulus as used in Betz et al. (2015).
    The authors used three stimuli with different grating frequencies. The three stimuli
    had a size of 10.2deg with 4 bars (lowSF) or 7.66deg with either 6 (mediumSF) or
    12 bars (highSF).

    Parameters
    ----------
    white_freq
        Str with grating frequency (options: high, medium, low)
    ppd
        Spatial resolution (pixels per degree).

    Returns
    -------
    stimulus
        2D array with White stimulus.

    """
    if white_freq == 'high':
        # High freq (0.8 cpd):
        bar_width = int(0.638 * ppd)
        n_bars = 12
    elif white_freq == 'medium':
        # Medium freq (0.4 cpd):
        bar_width = int(1.276 * ppd)
        n_bars = 6
    elif white_freq == 'low':
        # Low freq (0.2 cpd):
        bar_width = int(2.552 * ppd)
        n_bars = 4

    # Intensity values for grating bars
    grating_vals = [0.95, 1.05]
    target_val = 1.

    # Calculate position for gray patch and choose values
    target_pos = (n_bars / 2 * bar_width, (n_bars / 2 - .5) * bar_width)

    # Create square wave grating
    stimulus = np.ones((bar_width * n_bars, bar_width * n_bars)) * grating_vals[1]
    index = [i + j for i in range(bar_width) for j in range(0, bar_width * n_bars, bar_width * 2)]
    stimulus[index, :] = grating_vals[0]

    # Place test square at selected position
    y, x = target_pos
    stimulus[int(y):int(y+bar_width), int(x):int(x+bar_width)] = target_val
    return stimulus


def create_stimuli_dict(white_freq, params):
    """Convenience function for test case 1 that create one stimuli dict that
    can be fed to all different models.

    Parameters
    ----------
    white_freq
        Str with grating frequency (options: high, medium, low)
    params
        Parameters from parameters.py

    Returns
    -------
    stimuli_dict
        Dictionary with stimuli and some additional variables.

    """
    # Spatial resolution for simulations (pixels per degree)
    ppd = params.ppd

    # Visual extent of full stimulus (deg)
    visual_extent = params.visual_extent

    # Noisefrequencies of masks in cpd (Parameters taken from Betz2015)
    noisefreqs = params.noisefreqs
    n_masks = len(noisefreqs)

    # Number of trials:
    n_trials = params.n_trials

    # Create white stimulus in accordance to Betz2015
    white_stimulus = create_white_stimulus(white_freq, ppd)
    white_size = white_stimulus.shape[0]

    # Add gray background that covers the desired visual extent
    back_size = visual_extent[1]*2.*ppd-white_size
    white_stimulus = np.pad(white_stimulus, int(back_size/2.), 'constant', constant_values=(1.))
    stimulus_size = white_stimulus.shape[0]

    # Initiate nested list for stimuli.
    # We do this to be able to use n_trials same stimuli for all conditions
    stimuli_2d = [[] for i in range(n_trials)]

    for i in range(n_trials):
        stimuli = stimuli_2d[i]

        for j in range(n_masks):
            # Apply noise mask of given noisefreq
            stimuli.append(white_stimulus + create_noisemask(stimulus_size, noisefreqs[j], ppd))

    stimuli_dict = {'stimuli': stimuli_2d,
                    'stimulus_size': stimulus_size,
                    'background_size': back_size,
                    'white_freq': white_freq}
    return stimuli_dict


# %%
###############################
#           Plotting          #
###############################
# Convenience function for plotting results in experiment 1:
def plots_exp1(results_dict, plot_path: str):
    """Convenience function for plotting the results of test case 1.

    Parameters
    ----------
    results_dict
        Dictionary with results as created in main_case1.py
    plot_path
        Path to save the results plot.

    """
    vextent = results_dict['visual_extent']
    noisefreqs = results_dict['noisefreqs']
    n_masks = len(noisefreqs)

    # Plot stimuli from the last trial:
    stimuli = results_dict['stimuli']

    # Calculate average edge quality over trials
    corrs_mean = np.mean(results_dict['corrs_trials'], 0)
    corrs_std = np.std(results_dict['corrs_trials'], 0)

    fig = plt.figure(figsize=(24, 12))
    outer = matplotlib.gridspec.GridSpec(3, 1, wspace=0.4, hspace=0.4)

    # Plot input stimuli
    inner = matplotlib.gridspec.GridSpecFromSubplotSpec(1, n_masks, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    for i in range(n_masks):
        smin, smax = stimuli[i].min(), stimuli[i].max()
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(stimuli[i], cmap='gray', vmin=smin, vmax=smax, extent=vextent)
        ax.set_title('Noise: ' + str(noisefreqs[i]) + ' cpd')
        fig.add_subplot(ax)

    # Plot model output
    inner = matplotlib.gridspec.GridSpecFromSubplotSpec(1, n_masks, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    for i in range(n_masks):
        smin, smax = stimuli[i].min(), stimuli[i].max()
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(results_dict['model_outputs'][:, :, i], cmap='pink', extent=vextent)
        fig.add_subplot(ax)

    # Plot model performance
    inner = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.1, hspace=0.1)
    ax = plt.Subplot(fig, inner[1])
    ax.errorbar(noisefreqs, corrs_mean, yerr=corrs_std, marker='.', capsize=3)
    ax.set_title('Edge detection performance')
    ax.set(ylabel='Correlation with gt', xlabel='Noise center freq (cpd)', xscale='log',
           xticks=(0.58, 1, 9), xticklabels=(0.58, 1, 9), ylim=(0., 1.))
    fig.add_subplot(ax)
    plt.savefig(plot_path + '.png', dpi=100)
    plt.close()


# %%
###############################
#      Quantifying edges      #
###############################
def create_white_gt(imsize, back_size, white_freq, ppd, thickness):
    """Function to create ground truth edge template for White stimuli used test case 1.

    Parameters
    ----------
    imSize
        Size of the template.
    backsize
        Amount of padding that needs to be added.
    white_freq
        Str with grating frequency (options: high, medium, low).
    ppd
        Spatial resolution (pixels per degree).
    thickness
        Number of pixels by which the edge thickness should be increased.

    Returns
    -------
    gt
        Ground truth edge template.

    """
    t = int(thickness / 2)
    half = int(back_size / 2)
    mag_grating = 0.05
    mag_target = 0.025

    # Parameters from Betz et al (2015):
    if white_freq == 'high':
        # High freq (0.8 cpd):
        bar_width = int(0.638 * ppd)
        n_bars = 12
    elif white_freq == 'medium':
        # Medium freq (0.4 cpd):
        bar_width = int(1.276 * ppd)
        n_bars = 6
    elif white_freq == 'low':
        # Low freq (0.2 cpd):
        bar_width = int(2.552 * ppd)
        n_bars = 4

    # Initiate the template
    gt = np.zeros([imsize, imsize])

    # Add horizontal grating edges:
    idx = [i+j-t for i in range(half+bar_width, imsize-half, bar_width) for j in range(0, t*2)]
    gt[idx, half-t:imsize-half+t] = mag_grating

    # Add outer contour edges of stimulus:
    gt[half-t:imsize-half+t, half-t:half+t] = mag_target
    gt[half-t:imsize-half+t, imsize-half-t:imsize-half+t] = mag_target
    gt[half-t:half+t, half-t:imsize-half+t] = mag_target
    gt[imsize-half-t:imsize-half+t, half-t:imsize-half+t] = mag_target

    # Calculate position for gray patch
    y = n_bars / 2 * bar_width + half
    x = (n_bars / 2 - .5) * bar_width + half

    # Add edge signals for target with chosen thickness:
    gt[int(y-t):int(y+t+bar_width), int(x-t):int(x+t)] = mag_target
    gt[int(y-t):int(y+t+bar_width), int(x-t+bar_width):int(x+t+bar_width)] = mag_target
    gt[int(y-t):int(y+t), int(x-t):int(x+t+bar_width)] = mag_target
    gt[int(y-t+bar_width):int(y+t+bar_width), int(x-t):int(x+t+bar_width)] = mag_target

    # Normalize between 0 and 1
    gt = np.abs(gt) / np.abs(gt).max()
    return gt


def quantify_edges(array, template):
    """Function that quantifies the quality of edges given a template as Pearson
    correlation between an output and the template after aligning the output and
    template to produce maximal cross-correlation.

    Parameters
    ----------
    array
        Array with edges. The first two dimensions are the image dimensions that will be
        correlated with the ground truth image. The array can have two additional dimensions.
    template
        Ground truth edge template.

    Returns
    -------
    corrs
        Correlation(s) between output and ground truth template. The output will have
        two dimensions less than the input array

    """
    if len(array.shape) == 2:
        array0 = array
    elif len(array.shape) == 3:
        array0 = array[:, :, 0]
    else:
        array0 = array[:, :, 0, 0]

    pfac = 10
    x = np.arange(-pfac, pfac+1)
    array_pad = np.pad(array0, pfac)
    xcorr = signal.correlate2d(array_pad, np.squeeze(template), 'valid')
    idx_max = np.where(xcorr == xcorr.max())
    yshift = x[idx_max[0][0]]
    xshift = x[idx_max[1][0]]
    ystart = pfac + yshift
    yend = pfac + yshift + np.size(array, 0)
    xstart = pfac + xshift
    xend = pfac + xshift + np.size(array, 1)

    # Flatten template vector
    template_1d = template.flatten()

    if len(array.shape) == 2:
        array_pad = np.pad(array, pfac)
        array_shift = array_pad[ystart:yend, xstart:xend]
        array_1d = array_shift.flatten()
        corrs = np.corrcoef(template_1d, array_1d)[0, 1]

    elif len(array.shape) == 3:
        n_masks = np.size(array, -1)
        corrs = np.zeros(n_masks)
        for i in range(n_masks):
            array_pad = np.pad(array[:, :, i], pfac)
            array_shift = array_pad[ystart:yend, xstart:xend]
            array_1d = array_shift.flatten()
            corrs[i] = np.corrcoef(template_1d, array_1d)[0, 1]

    else:
        n_masks = np.size(array, -1)
        n_filters = np.size(array, -2)
        corrs = np.zeros([n_filters, n_masks])
        for i in range(n_masks):
            for j in range(n_filters):
                array_pad = np.pad(array[:, :, j, i], pfac)
                array_shift = array_pad[ystart:yend, xstart:xend]
                array_1d = array_shift.flatten()
                corrs[j, i] = np.corrcoef(template_1d, array_1d)[0, 1]
    return corrs


# %%
###############################
#     Run model / controls    #
###############################
def run_active_model(stimuli, drift, sfilts, tfilt, rb, integrate='mean2', norm=True):
    """Convenience function that runs our model(s) on the input stimuli.

    Parameters
    ----------
    stimuli
        List with input stimuli.
    drift
        2D discretized drift array (unit: px)
    sfilts
        List of spatial filters defined in frequency space. For no spatial filtering, use [1.]
    tfilt
        Temporal filter defined in frequency space. For no temporal filtering, use 1.
    rb
        Number of pixels removed from the image boundaries.
    integrate
        Integration method (options: mean2 or var).
    norm
        Bool. If True, use spatial scale specific normalization.

    Returns
    -------
    output
        List of model outputs.

    """
    n_masks, n_filters = len(stimuli), len(sfilts)

    # Initiate measures:
    nX = stimuli[0].shape[0]
    output = np.zeros([nX, nX, n_filters, n_masks])

    # Perform simulations:
    for k in range(n_masks):
        # Apply drift to stimulus
        stimulus_video = apply_drift(stimuli[k], drift, back_lum=stimuli[0].mean())

        # Calculate fft and fftshift:
        gfft_3d = np.fft.fftshift(fft.fftn(stimulus_video))

        for i in range(n_filters):
            # Perform spatiotemporal filtering:
            output_temp = gfft_3d * sfilts[i] * tfilt
            output_temp = fft.ifftn(np.fft.ifftshift(output_temp))

            if integrate == 'var':
                # Integrate over time using variance
                output[:, :, i, k] = np.real(output_temp).var(2)
            elif integrate == 'mean2':
                # Integrate over time using squared mean
                output[:, :, i, k] = np.real(output_temp**2.).mean(2)
            else:
                raise ValueError('integration needs to be mean2 or var')

    # Remove borders before normalization to avoid border effects
    output = remove_borders(output, rb)

    if norm:
        # Normalize by global mean at each spatial scale:
        output = output / np.expand_dims(output.mean(axis=(0, 1)), (0, 1))

    # Sum model responses over spatial scales:
    output = np.sum(output, 2)
    return output


# Run Canny edge detection on the stimuli
def run_canny(stimuli, lthresh, htresh, gkernel=3):
    """Convenience function that runs a Canny edge detector with given parameters.

    Parameters
    ----------
    stimuli
        List with input stimuli.
    lthresh
        Lower threshold for hysteresis procedure in Canny.
    htresh
        Upper threshold for hysteresis procedure in Canny.
    gkernel
        Gaussian kernel size for blurring the image.

    Returns
    -------
    output
        List of model outputs.

    """
    n_masks = len(stimuli)
    output = np.zeros(np.concatenate((stimuli[0].shape, np.array([n_masks])), axis=None))

    for i in range(n_masks):
        # Apply Gaussian blur and use Canny
        stimulus = (stimuli[i] / stimuli[i].max() * 255.).astype(np.uint8)
        stimulus = cv2.GaussianBlur(stimulus, (gkernel, gkernel), 0)
        output[:, :, i] = cv2.Canny(stimulus, lthresh, htresh)
    return output

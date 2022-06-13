"""
Script that reads in results for test case 1 from pickle files and visualizes them.

Last update on 08.06.2022
@author: lynnschmittwilken
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Let's read in params and functions from simulations:
sim_path = '../../simulations/'
sys.path.append(sim_path)
import parameters as params


#########################################
#           Helper functions            #
#########################################
def read_data(filepath: str):
    """Helper function to read results from pickle files.

    Parameters
    ----------
    filepath
        Path to pickle file.

    Returns
    -------
    data
        Dictionary with results of individual models.

    """
    with open(filepath + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data


def plot_exp_results(exp_data, g_idx: int, result_folder: str):
    """Helper function to plot experimental data from Betz et al. (2015).

    Parameters
    ----------
    exp_data
        Dict with experimental results.
    g_idx
        Index of grating (options: 0, 1, 2).
    result_folder
        Name of folder in which to save the plot.

    """
    noisefreqs = exp_data['noise_freqs']
    exp_means = exp_data['noise_eff'].mean(1)[g_idx, :]
    n_sqrt = np.sqrt(exp_data['noise_eff'].shape[1])
    exp_error = exp_data['noise_eff'].std(1)[g_idx, :] / n_sqrt

    # Create figure and plot results
    plt.figure(figsize=(3, 2))
    plt.plot(exp_data['grating_freqs'][g_idx], 0, 'r*', mec='r', ms=12)
    plt.errorbar(noisefreqs, exp_means, exp_error, marker='o', capsize=3.)
    plt.ylim(-3, 7)
    plt.xlim(0.2)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(exp_data['baseline_eff'][g_idx, :].mean(), linestyle='--')
    plt.xscale('log')
    plt.xticks(np.insert(noisefreqs, 0, 0.2), np.insert(noisefreqs, 0, 0.2))
    plt.savefig(result_folder + 'exp_results_grating' + str(g_idx) + '.png', dpi=300)
    plt.close()


def plot_sim_results(exp_data, g_idx: int, sim_data, result_folder: str):
    """Helper function to plot modeling results.

    Parameters
    ----------
    exp_data
        Dict with experimental results.
    g_idx
        Index of grating (options: 0, 1, 2).
    sim_data
        Dict with modeling results
    result_folder
        Name of folder in which to save the plot.

    """
    noisefreqs = exp_data['noise_freqs']

    # Spatiotemporal filtering followed by squared mean and global normalization
    m = 0
    n_sqrt = np.sqrt(sim_data[m]['n_trials'])
    m_means = sim_data[m]['corrs_trials'].mean(0)
    m_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Spatial filtering followed by squared mean and global normalization
    m = 1
    c1_means = sim_data[m]['corrs_trials'].mean(0)
    c1_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Spatiotemporal filtering followed by squared mean
    m = 3
    c2_means = sim_data[m]['corrs_trials'].mean(0)
    c2_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Temporal filtering followed by squared mean
    m = 2
    c3_means = sim_data[m]['corrs_trials'].mean(0)
    c3_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Spatial filtering followed by variance and global normalization
    m = 4
    v1_means = sim_data[m]['corrs_trials'].mean(0)
    v1_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Spatial filtering followed by variance
    m = 5
    v2_means = sim_data[m]['corrs_trials'].mean(0)
    v2_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # No filtering followed by variance
    m = 6
    v3_means = sim_data[m]['corrs_trials'].mean(0)
    v3_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Canny
    m = 7
    canny_means = sim_data[m]['corrs_trials'].mean(0)
    canny_error = sim_data[m]['corrs_trials'].std(0) / n_sqrt

    # Create figure and plot results
    cap = 3.

    plt.figure(figsize=(3, 2))
    plt.plot(exp_data['grating_freqs'][g_idx], 0, 'r*', mec='r', ms=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.errorbar(noisefreqs, m_means, m_error, c='C0', marker='o', capsize=cap, label='ST-M-N*')
    plt.errorbar(noisefreqs, c1_means, c1_error, c='C1', marker='o', capsize=cap, label='S-M-N')
    plt.errorbar(noisefreqs, c2_means, c2_error, c='C2', marker='o', capsize=cap, label='ST-M')
    plt.errorbar(noisefreqs, c3_means, c3_error, c='C4', marker='o', capsize=cap, label='T-M')
    plt.errorbar(noisefreqs, v1_means, v1_error, c='C5', marker='o', capsize=cap, label='S-V-N*')
    plt.errorbar(noisefreqs, v2_means, v2_error, c='C6', marker='o', capsize=cap, label='S-V')
    plt.errorbar(noisefreqs, v3_means, v3_error, c='C7', marker='o', capsize=cap, label='V')
    plt.errorbar(noisefreqs, canny_means, canny_error, c='C8', marker='o', capsize=cap, label='Canny')

    plt.ylim(-0.1, 1.)
    plt.xlim(0.2)
    plt.xscale('log')
    plt.xticks(np.insert(noisefreqs, 0, 0.2), np.insert(noisefreqs, 0, 0.2))
    plt.savefig(result_folder + 'sim_results_grating' + str(g_idx) + '.png', dpi=300)
    plt.close()


#########################################
#            Read-in results            #
#########################################
exp_folder = '../../databases/betz2015_data/'
sim_folder = sim_path + params.results_path1

# Experimental conditions for model and controls:
# Testing individual model components:
ST_M_N = 'ST_M_N'
S_M_N = 'S_M_N'
ST_M = 'ST_M'
T_M = 'T_M'

# Exchanging temporal filtering by variance
S_V_N = 'S_V_N'
S_V = 'S_V'
V = 'V'

# Canny:
canny = 'canny'

# Some re-occuring parameters for reading in data:
smax = '_' + str(params.smax)
D = '_D' + str(np.round(params.D*60.**2.))
T = '_T' + str(params.T)
ppd = '_ppd' + str(params.ppd)

lw = '_low'
md = '_medium'
hg = '_high'


##################### Experimental data #####################
# Read-in results from psychophysical experiments:
with open(exp_folder + 'betz2015_data.pickle', 'rb') as handle:
    exp_data = pickle.load(handle)


#################### Main model #####################
# Spatiotemporal filtering followed by squared mean and global normalization
ST_M_N_low = read_data(sim_folder + ST_M_N + smax + D + T + ppd + lw)
ST_M_N_med = read_data(sim_folder + ST_M_N + smax + D + T + ppd + md)
ST_M_N_hig = read_data(sim_folder + ST_M_N + smax + D + T + ppd + hg)


#################### Control models #####################
# Spatial filtering followed by squared mean and global normalization
S_M_N_low = read_data(sim_folder + S_M_N + smax + D + T + ppd + lw)
S_M_N_med = read_data(sim_folder + S_M_N + smax + D + T + ppd + md)
S_M_N_hig = read_data(sim_folder + S_M_N + smax + D + T + ppd + hg)

# Spatiotemporal filtering followed by squared mean and no normalization
ST_M_low = read_data(sim_folder + ST_M + smax + D + T + ppd + lw)
ST_M_med = read_data(sim_folder + ST_M + smax + D + T + ppd + md)
ST_M_hig = read_data(sim_folder + ST_M + smax + D + T + ppd + hg)

# Temporal filtering followed by squared mean and no normalization
T_M_low = read_data(sim_folder + T_M + smax + D + T + ppd + lw)
T_M_med = read_data(sim_folder + T_M + smax + D + T + ppd + md)
T_M_hig = read_data(sim_folder + T_M + smax + D + T + ppd + hg)


#################### Var models #####################
# Spatial filtering followed by variance and no normalization
S_V_N_low = read_data(sim_folder + S_V_N + smax + D + T + ppd + lw)
S_V_N_med = read_data(sim_folder + S_V_N + smax + D + T + ppd + md)
S_V_N_hig = read_data(sim_folder + S_V_N + smax + D + T + ppd + hg)

# Spatial filtering followed by variance and no normalization
S_V_low = read_data(sim_folder + S_V + smax + D + T + ppd + lw)
S_V_med = read_data(sim_folder + S_V + smax + D + T + ppd + md)
S_V_hig = read_data(sim_folder + S_V + smax + D + T + ppd + hg)

# No filtering followed by variance
V_low = read_data(sim_folder + V + smax + D + T + ppd + lw)
V_med = read_data(sim_folder + V + smax + D + T + ppd + md)
V_hig = read_data(sim_folder + V + smax + D + T + ppd + hg)


#################### Canny #####################
canny_low = read_data(sim_folder + canny + smax + D + T + ppd + lw)
canny_med = read_data(sim_folder + canny + smax + D + T + ppd + md)
canny_hig = read_data(sim_folder + canny + smax + D + T + ppd + hg)


# Make list of all model data for lf grating:
lf_model_data = [ST_M_N_low,
                 S_M_N_low,
                 T_M_low,
                 ST_M_low,
                 S_V_N_low,
                 S_V_low,
                 V_low,
                 canny_low]

# Make list of all model data for mf grating:
mf_model_data = [ST_M_N_med,
                 S_M_N_med,
                 T_M_med,
                 ST_M_med,
                 S_V_N_med,
                 S_V_med,
                 V_med,
                 canny_med]

# Make list of all model data for hf grating:
hf_model_data = [ST_M_N_hig,
                 S_M_N_hig,
                 T_M_hig,
                 ST_M_hig,
                 S_V_N_hig,
                 S_V_hig,
                 V_hig,
                 canny_hig]


#########################################
#               Plotting                #
#########################################
# Create output folder:
result_folder = 'quantitative_results/'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)


plotting_exp = True

if plotting_exp:
    # Plot experimental results for LSF grating
    plot_exp_results(exp_data, 0, result_folder)
    # Plot experimental results for MSF grating
    plot_exp_results(exp_data, 1, result_folder)
    # Plot experimental results for HSF grating
    plot_exp_results(exp_data, 2, result_folder)

plotting_sim = True

if plotting_sim:
    # Plot experimental results for LSF grating
    plot_sim_results(exp_data, 0, lf_model_data, result_folder)
    # Plot experimental results for MSF grating
    plot_sim_results(exp_data, 1, mf_model_data, result_folder)
    # Plot experimental results for HSF grating
    plot_sim_results(exp_data, 2, hf_model_data, result_folder)

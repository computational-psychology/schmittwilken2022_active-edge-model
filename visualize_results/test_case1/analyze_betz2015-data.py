"""
Read all the experimental data from the csv files provided in Betz2015.
Then calculate the illusion strength at baseline and the illusion strength
for the noisemasking stimuli, and save these outputs in a pickle file

Last update on 08.06.2022
@author: lynnschmittwilken
"""

import os
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append('../../simulations')
import parameters as params


#########################################
#               Functions               #
#########################################
def get_data(datadir: str, vp_names):
    """Helper function to read individual data from experiments of Betz et al. (2015)
    into one large dataframe

    Parameters
    ----------
    datadir
        Path to data
    vp_names
        List with names of all participants

    Returns
    -------
    all_data
        Dataframe with all participants' data

    """
    all_data = []

    # Read in the experimental data from all the subfolders
    for vp in vp_names:
        grating_freqs = os.listdir(os.path.join(datadir, vp))
        for sf in grating_freqs:
            contrasts = os.listdir(os.path.join(datadir, vp, sf))
            for contrast in contrasts:
                filenames = os.listdir(os.path.join(datadir, vp, sf, contrast))
                for filename in filenames:
                    data = pd.read_csv(os.path.join(datadir, vp, sf, contrast, filename), sep=' ')

                    # Add a column for the subject ID (here: ranging from 0 to 10)
                    vp_id = np.repeat(vp, data.shape[0])
                    data['vp'] = vp_id

                    # Add the individual data to the full dataset
                    all_data.append(data)

    # Concat dataframes of all subjects:
    all_data = pd.concat(all_data)

    # Let's reset the indices of the full dataframe, so it ranges from 0 to n
    all_data = all_data.reset_index(drop=True)

    # Convert luminance encoding to cd/m**2
    all_data['match_lum'] = all_data['match_lum'] * 88.
    all_data['match_initial'] = all_data['match_initial'] * 88.
    all_data['test_lum'] = all_data['test_lum'] * 88.
    return all_data


def compute_baseline_effect(data):
    """Function to compute the magnitude of the lightness effect per repetition for the
    baseline (no noise) for all three grating frequencies

    Parameters
    ----------
    data
        Dataframe with all participants' data

    Returns
    -------
    baseline_eff_2d
        Nested list with effect magnitudes per repetition for each grating frequency

    """
    # We will calculate the performance for each grating:
    gfs = np.unique(data['grating_freq'])
    n_gratings = len(gfs)
    n_vp = len(vp_names)

    # Initiate nested list for baseline effect
    baseline_eff_2d = [[] for i in range(n_gratings)]

    # Calculate effects for each grating, each subjects (and each noisefreq):
    for g in range(n_gratings):
        # Get data for each grating freq
        data_by_gf = data[data['grating_freq'] == gfs[g]]

        # Baseline data without noise:
        baseline_data = data_by_gf[data_by_gf['noise_type'] == 'none']

        baseline_eff = baseline_eff_2d[g]

        for k in range(n_vp):
            # Compute baseline effectsize
            baseline_vp = baseline_data[baseline_data['vp'] == vp_names[k]]
            baseline_inc_data = baseline_vp[baseline_vp['coaxial_lum'] == -1]
            baseline_inc = baseline_inc_data['match_lum']
            baseline_dec_data = baseline_vp[baseline_vp['coaxial_lum'] == 1]
            baseline_dec = baseline_dec_data['match_lum']
            baseline_effect = baseline_inc.mean() - baseline_dec.mean()
            baseline_eff.append(baseline_effect)
    return baseline_eff_2d


def compute_noise_effect(data):
    """Function to compute the magnitude of the lightness effect per repetition for the
    all noise conditions and for all three grating frequencies

    Parameters
    ----------
    data
        Dataframe with all participants' data

    Returns
    -------
    noise_eff_2d
        Nested list with effect magnitudes per repetition for each grating frequency

    """
    # We will calculate the average performance for each grating:
    gfs = np.unique(data['grating_freq'])
    n_gratings = len(gfs)
    n_vp = len(vp_names)

    # Initiate nested list for noise data effect
    noise_eff_2d = [[] for i in range(n_gratings)]

    # Calculate effects for each grating, each subjects (and each noisefreq):
    for g in range(n_gratings):
        # Get data for each grating freq
        data_by_gf = data[data['grating_freq'] == gfs[g]]

        # Experimental data for noise stimuli:
        noise_data = data_by_gf[data_by_gf['noise_type'] == 'global']
        noise_eff = noise_eff_2d[g]

        for k in range(n_vp):
            # Compute effect for each noise stimulus
            noise_vp = noise_data[noise_data['vp'] == vp_names[k]]
            noise_freqs = np.unique(noise_vp['noise_freq'])
            n_freqs = len(noise_freqs)
            n_reps = len(np.unique(noise_vp['rep']))

            illusion_strength = np.empty((n_freqs, n_reps))

            for i in range(n_freqs):
                data_by_nf = noise_vp[noise_vp['noise_freq'] == noise_freqs[i]]

                # Get data for increments
                data_increments = data_by_nf[data_by_nf['coaxial_lum'] == -1]
                data_increments = data_increments.sort_values(by=['rep'])
                inc = data_increments['match_lum']

                # Get data for decrements
                data_decrements = data_by_nf[data_by_nf['coaxial_lum'] == 1]
                data_decrements = data_decrements.sort_values(by=['rep'])
                dec = data_decrements['match_lum']

                # Compute difference:
                illusion_strength[i, :] = np.asarray(inc).flatten() - np.asarray(dec).flatten()

            noise_eff.append(illusion_strength)
    return noise_eff_2d


#########################################
#                 Main                  #
#########################################
datadir = '../../databases/betz2015_data/'

# Read vp names from folder names in datadir
vp_names = [name for name in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, name))]

# Load all data
data = get_data(datadir, vp_names)

# Calculate baseline effect pd:
baseline_eff_list = compute_baseline_effect(data)
noise_eff_list = compute_noise_effect(data)

grating_freqs = np.unique(data['grating_freq'])
n_gratings = len(grating_freqs)
n_vp = len(vp_names)
noise_freqs = np.unique(data['noise_freq'])
noise_freqs = noise_freqs[noise_freqs != 0]
n_freqs = len(noise_freqs)

# Reorganize list contents in np array and average over reps:
baseline_eff = np.zeros([n_gratings, n_vp])
noise_eff = np.zeros([n_gratings, n_vp, n_freqs])

for i in range(n_gratings):
    baseline_eff[i, :] = baseline_eff_list[i]
    for j in range(n_vp):
        noise_eff[i, j, :] = noise_eff_list[i][j].mean(1)


#########################################
#             Save pickles              #
#########################################
save_dict = {'vp_names': vp_names,
             'grating_freqs': grating_freqs,
             'noise_freqs': noise_freqs,
             'baseline_eff': baseline_eff,
             'noise_eff': noise_eff}

pickle_file = 'betz2015_data.pickle'

with open(datadir + pickle_file, 'wb') as handle:
    pickle.dump(save_dict, handle)

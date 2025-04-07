# Imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import LogLocator, LogFormatter
from extract_channels import *  # Custom utility for handling channel data

def get_trial_arrays(featData, feats, rel_start, rel_end):
    """
    Extracts trial-aligned data slices from featData for a given feature
    and relative start/end frame offsets.
    """
    feats = featData[feats]
    goTrials = featData['goTrialEpochs']
    res_list = []
    for t in range(goTrials.shape[0]):
        trial = feats[(goTrials[t, 0] + rel_start):(goTrials[t, 0] + rel_end)]
        res_list.append(trial)
    return np.array(res_list)

def plot_trial_timecourse(featData, title='', rel_start=-20, rel_end=80, saveFig=False, dpi=300, log=False):
    """
    Plots the average spike band power over time for different brain regions
    alongside the audio envelope.
    """
    trials = get_trial_arrays(featData, 'spikePow', rel_start, rel_end)
    audio = get_trial_arrays(featData, 'audioEnvelope', rel_start, rel_end)

    # Compute means for different brain areas
    mean_44_inf = np.mean(extract_channel_data(trials, area_44_inferior, 2), axis=(0, 2))
    mean_44_sup = np.mean(extract_channel_data(trials, area_44_superior, 2), axis=(0, 2))
    mean_6v_inf = np.mean(extract_channel_data(trials, area_6v_superior, 2), axis=(0, 2))
    mean_6v_sup = np.mean(extract_channel_data(trials, area_6v_inferior, 2), axis=(0, 2))
    mean_audio = np.mean(audio, axis=(0, 2))

    times = 20 / 1000 * np.linspace(rel_start, rel_end, trials.shape[1])  # Convert frame indices to time

    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Average spike band power')
    if log:
        ax1.set_yscale('log')

    # Plot spike band power
    ax1.plot(times, mean_6v_inf, label='inf6v', color='blue')
    ax1.plot(times, mean_6v_sup, label='sup6v', color='navy')
    ax1.plot(times, mean_44_inf, label='inf44', color='limegreen')
    ax1.plot(times, mean_44_sup, label='sup44', color='darkgreen')

    # Create secondary y-axis for audio
    ax2 = ax1.twinx()
    ax2.set_ylabel('Microphone volume')
    if log:
        ax2.set_yscale('log')
    ax2.plot(times, mean_audio, label='audio', color='black', linewidth=3)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    if saveFig:
        plt.savefig(saveFig, dpi=dpi)
    plt.show()

def count_alternating_segments(arr):
    """
    Returns lengths of alternating segments of 0s and 1s in a binary array.
    """
    if len(arr) == 0:
        return np.array([]), np.array([])

    change_indices = np.where(np.diff(arr) != 0)[0] + 1
    segment_lengths = np.diff(np.concatenate(([0], change_indices, [len(arr)])))

    zeros_counts = segment_lengths[::2] if arr[0] == 0 else segment_lengths[1::2]
    ones_counts = segment_lengths[1::2] if arr[0] == 0 else segment_lengths[::2]

    return zeros_counts, ones_counts

def aggregate_trials(data):
    """
    Splits spike power data into separate lists of 0-state and 1-state trials
    based on the 'trialState' array.
    """
    trial_state_arr = data['trialState'][:, 0]
    change_indices = list(np.where(np.diff(trial_state_arr) != 0)[0] + 1)
    change_indices.insert(0, 0)
    change_indices.append(len(trial_state_arr))

    zero_list, one_list = [], []
    spikePowDat = data['spikePow']

    for num, idx in enumerate(change_indices[:-1]):
        trial = spikePowDat[idx:change_indices[num + 1]]
        if trial_state_arr[idx] == 0:
            zero_list.append(trial)
        else:
            one_list.append(trial)

    return zero_list, one_list

def avg_agg(agg_trials, axis=False):
    """
    Truncates all trials to the minimum length and returns their average.
    """
    trial_lengths = [trial.shape[0] for trial in agg_trials]
    min_len = np.min(trial_lengths)
    agg_trial_np = np.array([trial[:min_len] for trial in agg_trials])
    if axis:
        agg_trial_np = np.mean(agg_trial_np, axis=axis)
    return agg_trial_np

def calc_trial_mean(data):
    """
    Calculates the average spike power for each trial segment and separates
    them based on the trial state (0 or 1).
    """
    trial_state_arr = data['trialState'][:, 0]
    change_indices = list(np.where(np.diff(trial_state_arr) != 0)[0] + 1)
    change_indices.insert(0, 0)
    change_indices.append(len(trial_state_arr))

    zero_list, one_list = [], []
    spikePowDat = data['spikePow']

    for num, idx in enumerate(change_indices[:-1]):
        trial_avg = np.mean(spikePowDat[idx:change_indices[num + 1]], axis=0)
        if trial_state_arr[idx] == 0:
            zero_list.append(trial_avg)
        else:
            one_list.append(trial_avg)

    return np.array(zero_list), np.array(one_list)

def calculate_bins(arrays, num_bins):
    """
    Calculates histogram bin edges spanning the combined range of input arrays.
    """
    combined_data = np.concatenate(arrays)
    min_val, max_val = np.min(combined_data), np.max(combined_data)
    return np.linspace(min_val, max_val, num_bins + 1)

def plot_bandpower_histograms(chan_avg_arrays, title, labels, colors, num_bins=20,
                               figsize=(10, 6), alpha=0.5, xlim=False, saveFig=False, dpi=300):
    """
    Plots overlaid histograms of average bandpower for different trial types or groups.
    """
    bins = calculate_bins(chan_avg_arrays, num_bins)
    plt.figure(figsize=figsize)
    for i, arr in enumerate(chan_avg_arrays):
        sns.histplot(arr, bins=bins, kde=True, color=colors[i], label=labels[i], alpha=alpha)
    plt.xlabel("Average bandpower within period")
    plt.ylabel("Channel count")
    if xlim:
        plt.xlim(0, xlim)
    plt.legend()
    plt.title(title)
    if saveFig:
        plt.savefig(saveFig, dpi=dpi)
    plt.show()

def plot_array_activity(data, area, title=''):
    """
    Plots the average activity across channels in a brain area as an 8x8 heatmap.
    """
    avg = np.mean(extract_channel_data(data, area), axis=0)
    avg_reshaped = list_to_8x8_array(avg)
    labels = list_to_8x8_array(area)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(avg_reshaped, cmap='viridis', interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax.set_title(title)

    # Label each square with channel name/index
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(labels[i, j]), ha='center', va='center', fontsize=10, color='white')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def plot_js_heatmap_lower_triangle(chan_avg_arrays, title='', labels=None, saveFig=False):
    """
    Plots a lower triangular matrix of Jensen-Shannon divergences between distributions.
    """
    # Normalize arrays to sum to 1 for valid probability distributions
    normalized = [arr / np.sum(arr) for arr in chan_avg_arrays]

    n = len(chan_avg_arrays)
    js_matrix = np.zeros((n, n))

    # Compute pairwise JS divergence (symmetric)
    for i in range(n):
        for j in range(n):
            js = jensenshannon(normalized[i], normalized[j]) ** 2
            js_matrix[i, j] = js

    # Mask upper triangle for plotting
    mask = np.triu(np.ones_like(js_matrix, dtype=bool), k=1)

    if labels is None:
        labels = [f'Dist {i}' for i in range(n)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(js_matrix, mask=mask, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".3f", cmap='viridis', square=True, cbar_kws={'label': 'JS Divergence'})
    plt.title(title)
    plt.tight_layout()
    if saveFig:
        plt.savefig(saveFig, dpi=300)
    plt.show()

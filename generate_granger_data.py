import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import json
from tqdm import tqdm
from scipy.signal import correlate, butter, filtfilt
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed
from itertools import product
import warnings
from extract_channels import *

# Specify the directory containing the tuningTasks files, and 
# the directory to save granger causality results.
gc_numpy_dir='/home/aidan/data/granger_numpy/'
dataDir = '/home/aidan/data/'
fiftyWordDat = sio.loadmat(dataDir+'tuningTasks/t12.2022.05.03_fiftyWordSet.mat')

def get_all_pval(granger_causality_result):
    pval_list = []
    for lag in range(1,len(granger_causality_result)+1):
        pval = granger_causality_result[lag][0]['ssr_ftest'][1]
        pval_list.append(pval)
    return pval_list

def process_single_trial(trial_num, cue, data, caused, causal, maxlag, speaking):
    go_start, go_end = data['goTrialEpochs'][trial_num]
    
    # Define segment length based on speaking condition
    expected_length = 40
    if speaking:
        caused_seg = caused[go_start-15:go_start+25, :]
        causal_seg = causal[go_start-15:go_start+25, :]
    else:
        caused_seg = caused[go_end:go_end+40, :]
        causal_seg = causal[go_end:go_end+40, :]
    
    # Check for insufficient time points
    if caused_seg.shape[0] < expected_length or causal_seg.shape[0] < expected_length:
        print(f"Skipping trial {trial_num} due to insufficient time points ({caused_seg.shape[0]} available).")
        return None  # Skip this trial

    if isinstance(maxlag, list):
        num_lags = len(maxlag)
    else:
        num_lags = maxlag

    cue_res = np.zeros((caused.shape[1], causal.shape[1], num_lags))

    for chan0, chan1 in product(range(caused.shape[1]), range(causal.shape[1])):
        cmp = np.stack([caused_seg[:, chan0], causal_seg[:, chan1]], axis=1)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress UserWarnings
                gc_res = grangercausalitytests(cmp, maxlag=maxlag, verbose=False)
            
            result = get_all_pval(gc_res)
            if len(result) != num_lags:
                print(f"Shape mismatch in trial {trial_num}: result has {len(result)} lags, expected {num_lags}.")
            
            cue_res[chan0, chan1] = result
        except Exception as e:
            print(f"Error in trial {trial_num}, channels {chan0}-{chan1}: {e}")
            cue_res[chan0, chan1] = [-1, -1]
    
    return cue, cue_res

def generate_multiarray_results(data, caused_chans, causal_chans, maxlag, speaking, data_type='spikePow', n_jobs=1):
    caused = extract_channel_data(data[data_type], caused_chans)
    causal = extract_channel_data(data[data_type], causal_chans)

    processed_results = Parallel(n_jobs=n_jobs)(delayed(process_single_trial)(
        trial_num, data['trialCues'][trial_num, 0], data, caused, causal, maxlag, speaking
    ) for trial_num in tqdm(range(len(data['trialCues']))))

    # Remove None values (skipped trials)
    processed_results = [res for res in processed_results if res is not None]

    # Maintain order of first appearance (not sorting)
    cues = list(dict.fromkeys(cue for cue, _ in processed_results))  # Keeps order

    results_dict = {cue: [] for cue in cues}
    
    for cue, cue_res in processed_results:
        results_dict[cue].append(cue_res)
    
    # Convert list of lists to a numpy object array
    results_np = np.array([np.array(results_dict[cue]) for cue in cues], dtype=object)

    return cues, results_np


def save_multiarray_results(cues, results_np, gc_numpy_dir='/home/aidan/data/granger_numpy/', cues_filename='cues.json', results_filename='data.npy'):
    # Define output directory
    os.makedirs(gc_numpy_dir, exist_ok=True)
    
    # Convert cues to Python integers (fix JSON serialization issue)
    cues_list = [int(cue) for cue in cues]  # Ensures JSON compatibility
    
    # Save cues as JSON
    cues_json_path = os.path.join(gc_numpy_dir, cues_filename)
    with open(cues_json_path, 'w') as f:
        json.dump(cues_list, f)
    print(f"Saved cues list to {cues_json_path}")

    # Determine the minimum number of trials that exist for all cues
    min_trials_per_cue = min([arr.shape[0] for arr in results_np])  # Find the shortest set of trials
    num_cues = len(cues)
    num_channels = results_np[0].shape[1]  # Should be 64
    num_lags = results_np[0].shape[3]      # Should be 12

    # Trim all cues to have exactly min_trials_per_cue trials
    trimmed_results = np.stack([arr[:min_trials_per_cue] for arr in results_np], axis=0)

    # Ensure the final array shape is as expected
    assert trimmed_results.shape == (num_cues, min_trials_per_cue, num_channels, num_channels, num_lags), \
        f"Unexpected shape: {trimmed_results.shape}, expected ({num_cues}, {min_trials_per_cue}, {num_channels}, {num_channels}, {num_lags})"

    # Save the final 5D numpy array
    npy_path = os.path.join(gc_numpy_dir, results_filename)
    np.save(npy_path, trimmed_results)
    print(f"Saved trimmed numpy array to {npy_path} with shape {trimmed_results.shape}")

    return

# Run Granger Causality tests for every pair of arrays (Area 44 -> 6v and reverse):

# Modify given your resources
n_jobs=-16

# Speaking condition

cues, sup44_gc_sup6v_results = generate_multiarray_results(fiftyWordDat, area_6v_superior, area_44_superior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, sup44_gc_sup6v_results, gc_numpy_dir,
                        cues_filename='sup44_gc_sup6v_cues.json', results_filename='sup44_gc_sup6v.npy')

cues, sup44_gc_inf6v_results = generate_multiarray_results(fiftyWordDat, area_6v_inferior, area_44_superior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, sup44_gc_inf6v_results, gc_numpy_dir,
                        cues_filename='sup44_gc_inf6v_cues.json', results_filename='sup44_gc_inf6v.npy')

cues, inf44_gc_sup6v_results = generate_multiarray_results(fiftyWordDat, area_6v_superior, area_44_inferior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, inf44_gc_sup6v_results, gc_numpy_dir,
                        cues_filename='inf44_gc_sup6v_cues.json', results_filename='inf44_gc_sup6v.npy')

cues, inf44_gc_inf6v_results = generate_multiarray_results(fiftyWordDat, area_6v_inferior, area_44_inferior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, inf44_gc_inf6v_results, gc_numpy_dir,
                        cues_filename='inf44_gc_inf6v_cues.json', results_filename='inf44_gc_inf6v.npy')

# Swapped brain areas
cues, sup6v_gc_sup44_results = generate_multiarray_results(fiftyWordDat, area_44_superior, area_6v_superior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, sup6v_gc_sup44_results, gc_numpy_dir,
                        cues_filename='sup6v_gc_sup44_cues.json', results_filename='sup6v_gc_sup44.npy')

cues, sup6v_gc_inf44_results = generate_multiarray_results(fiftyWordDat, area_44_inferior, area_6v_superior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, sup6v_gc_inf44_results, gc_numpy_dir,
                        cues_filename='sup6v_gc_inf44_cues.json', results_filename='sup6v_gc_inf44.npy')

cues, inf6v_gc_sup44_results = generate_multiarray_results(fiftyWordDat, area_44_superior, area_6v_inferior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, inf6v_gc_sup44_results, gc_numpy_dir,
                        cues_filename='inf6v_gc_sup44_cues.json', results_filename='inf6v_gc_sup44.npy')

cues, inf6v_gc_inf44_results = generate_multiarray_results(fiftyWordDat, area_44_inferior, area_6v_inferior, 12, speaking=True, n_jobs=n_jobs)
save_multiarray_results(cues, inf6v_gc_inf44_results, gc_numpy_dir,
                        cues_filename='inf6v_gc_inf44_cues.json', results_filename='inf6v_gc_inf44.npy')

# Non-speaking condition (control)
cues, ctrl_sup44_gc_sup6v_results = generate_multiarray_results(fiftyWordDat, area_6v_superior, area_44_superior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_sup44_gc_sup6v_results, gc_numpy_dir,
                        cues_filename='ctrl_sup44_gc_sup6v_cues.json', results_filename='ctrl_sup44_gc_sup6v.npy')

cues, ctrl_sup44_gc_inf6v_results = generate_multiarray_results(fiftyWordDat, area_6v_inferior, area_44_superior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_sup44_gc_inf6v_results, gc_numpy_dir=,
                        cues_filename='ctrl_sup44_gc_inf6v_cues.json', results_filename='ctrl_sup44_gc_inf6v.npy')

cues, ctrl_inf44_gc_sup6v_results = generate_multiarray_results(fiftyWordDat, area_6v_superior, area_44_inferior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_inf44_gc_sup6v_results, gc_numpy_dir=,
                        cues_filename='ctrl_inf44_gc_sup6v_cues.json', results_filename='ctrl_inf44_gc_sup6v.npy')

cues, ctrl_inf44_gc_inf6v_results = generate_multiarray_results(fiftyWordDat, area_6v_inferior, area_44_inferior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_inf44_gc_inf6v_results, gc_numpy_dir,
                        cues_filename='ctrl_inf44_gc_inf6v_cues.json', results_filename='ctrl_inf44_gc_inf6v.npy')

# Swapped brain areas
cues, ctrl_sup6v_gc_sup44_results = generate_multiarray_results(fiftyWordDat, area_44_superior, area_6v_superior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_sup6v_gc_sup44_results, gc_numpy_dir,
                        cues_filename='ctrl_sup6v_gc_sup44_cues.json', results_filename='ctrl_sup6v_gc_sup44.npy')

cues, ctrl_sup6v_gc_inf44_results = generate_multiarray_results(fiftyWordDat, area_44_inferior, area_6v_superior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_sup6v_gc_inf44_results, gc_numpy_dir,
                        cues_filename='ctrl_sup6v_gc_inf44_cues.json', results_filename='ctrl_sup6v_gc_inf44.npy')

cues, ctrl_inf6v_gc_sup44_results = generate_multiarray_results(fiftyWordDat, area_44_superior, area_6v_inferior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_inf6v_gc_sup44_results, gc_numpy_dir,
                        cues_filename='ctrl_inf6v_gc_sup44_cues.json', results_filename='ctrl_inf6v_gc_sup44.npy')

cues, ctrl_inf6v_gc_inf44_results = generate_multiarray_results(fiftyWordDat, area_44_inferior, area_6v_inferior, 12, speaking=False, n_jobs=n_jobs)
save_multiarray_results(cues, ctrl_inf6v_gc_inf44_results, gc_numpy_dir,
                        cues_filename='ctrl_inf6v_gc_inf44_cues.json', results_filename='ctrl_inf6v_gc_inf44.npy')


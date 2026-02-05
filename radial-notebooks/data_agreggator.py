#%%

import sys
import os

base_dir = r"C:\Users\acostamendoza\projects\distance-project"
data_dir = r"D:\data-distance-coding-project\processed-data"
sys.path.append(base_dir)
from src.data.RadialDevice import RadialDevice
from src.viz.SessionVisualizer import SessionVisualizer
from src.viz.CellVisualizer import CellVisualizer
from src.analysis.CellAnalysis import CellAnalyzer, SessionAnalyzer
from src.decoding.Decoders import MLE, MLEDistance, RidgeDistance
from src.decoding.NNDecoder import NNDecoder

import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.workflows import load_dataframe

mouse_name = 'SAM040'
experiment_name = 'tiling_120um'
base_data_dir = r'D:\data-distance-coding-project\processed-data'
multi_session_dir = r'D:\data-distance-coding-project\multi-session-data'

def build_traces_dict(mouse_name, experiment_name, base_data_dir=base_data_dir, reliable_only=True):
    """
    Load all traces for cells in multi_df with trial metadata.
    
    Returns:
        traces: dict {df_index: (n_repeats, n_positions) array}
        trial_info: dict {session_name: {'repeat_indices', 'distances', 'monocular'}}
    """
    multi_df = load_dataframe(mouse_name=mouse_name, experiment_name=experiment_name)
    viz = SessionVisualizer(multi_df, mouse_name=mouse_name, experiment_name=experiment_name)
    reliable_df = viz.filter_by_reliability(multi_df, reliable_type='reliable', threshold=0.85)
    # Filter for reliable cells if requested
    if reliable_only:
        df = reliable_df
        print(f"Filtered to {len(df)} reliable cells (from {len(multi_df)} total)")
    else:
        df = multi_df
    
    traces = {}
    trial_info = {}  # Store trial metadata per session
    
    for session_name, session_group in df.groupby('session_name'):
        print(f"Loading traces for {session_name} ({len(session_group)} cells)...")
        
        # Parse session_name
        parts = session_name.split('_')
        record_date = parts[0]
        mouse_name = parts[1]
        experiment_name = '_'.join(parts[2:])
        
        device = RadialDevice(
            mouse_name=mouse_name,
            record_date=record_date,
            experiment_name=experiment_name,
            base_data_dir=base_data_dir
        )
        
        # ALL trials (including controls)
        n_trials = len(device.distance_sequence)
        all_repeats = np.arange(n_trials)

        trial_info[session_name] = {
            'repeat_indices': all_repeats,
            'distances': device.distance_sequence.copy(),
            'monocular': device.monocular_sequence.copy()
        }

        # Get traces for each cell (all trials including controls)
        for idx, row in session_group.iterrows():
            cell_id = row['cell_id']
            cell_idx = np.where(device.suite2p_id == cell_id)[0][0]

            all_traces = []
            for repeat in all_repeats:
                on_trace = device.get_on_trace(cell_idx, repeat)
                all_traces.append(on_trace.T)
            traces[idx] = np.array(all_traces)
    
    return traces, trial_info, df

#%%
# Usage
traces, trial_info, reliable_df = build_traces_dict(
    mouse_name=mouse_name, experiment_name=experiment_name, base_data_dir=base_data_dir, reliable_only=True)

# Now you can relate traces to conditions:
# For a cell at index `idx` from session `session_name`:
#   traces[idx][i] corresponds to:
#     - distance: trial_info[session_name]['distances'][i]
#     - monocular: trial_info[session_name]['monocular'][i]

# # Get control trials (distance==4) for a cell
# session = reliable_df.loc[idx, 'session_name']
# control_mask = trial_info[session]['distances'] == 4
# control_traces = traces[idx][control_mask]

# # Get near + monocular
# near_mono_mask = (trial_info[session]['distances'] == 0) & (trial_info[session]['monocular'] == 1)

#%%
cache_path = os.path.join(multi_session_dir, f'SAM040_tiling_120um_traces.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump({'traces': traces, 'trial_info': trial_info, 'df': reliable_df}, f)
print(f"Saved {len(traces)} cells with traces to {cache_path}")

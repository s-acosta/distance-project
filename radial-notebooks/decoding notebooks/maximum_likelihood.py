"""
Non-parametric Maximum Likelihood Decoder
==========================================

Uses empirical mean responses (no Gaussian tuning curve fitting) with
Gaussian noise model for likelihood computation.
"""

import sys
import os
base_dir = r"C:\Users\acostamendoza\projects\distance-project"
data_dir = r"D:\data-distance-coding-project\processed-data"
sys.path.append(base_dir)
from src.data.RadialDevice import RadialDevice
from src.viz.SessionVisualizer import SessionVisualizer
from src.analysis.CellAnalysis import SessionAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
from src.workflows import load_dataframe


# =============================================================================
# DATA LOADING
# =============================================================================

mouse_name = 'SAM041'
record_date = '251128'
experiment_name = 'tiling_3_120um'

device = RadialDevice(mouse_name = mouse_name, record_date = record_date, experiment_name = experiment_name)
session_df = load_dataframe(mouse_name = mouse_name, record_date = record_date, experiment_name = experiment_name)
viz = SessionVisualizer(session_df, animal_id = mouse_name, experiment_name = experiment_name)

# =============================================================================
# NON-PARAMETRIC MLE DECODER
# =============================================================================

def compute_empirical_tuning(cell, train_trials):
    """
    Compute empirical mean and variance of response at each position.

    Parameters
    ----------
    cell : cell ID
    train_trials : array of trial indices to use for training

    Returns
    -------
    means : array (n_positions,) - mean response at each position
    variances : array (n_positions,) - variance at each position
    """
    # Get responses for all training trials
    # Each trial gives a response trace of shape (n_positions,)
    trial_responses = np.array([
        device.get_repeat_trace(cell, trial_idx)
        for trial_idx in train_trials
    ])  # Shape: (n_trials, n_positions)

    # Compute mean and variance across trials at each position
    means = np.mean(trial_responses, axis=0)
    variances = np.var(trial_responses, axis=0)

    return means, variances


def compute_all_neuron_stats(cells_to_include, train_trials):
    """
    Compute empirical tuning statistics for all neurons.

    Returns
    -------
    neuron_means : array (n_neurons, n_positions)
    neuron_variances : array (n_neurons, n_positions)
    """
    n_neurons = len(cells_to_include)
    n_positions = len(device.angle_range)

    neuron_means = np.zeros((n_neurons, n_positions))
    neuron_variances = np.zeros((n_neurons, n_positions))

    for i, cell in enumerate(cells_to_include):
        means, variances = compute_empirical_tuning(cell, train_trials)
        neuron_means[i] = means
        neuron_variances[i] = variances

    return neuron_means, neuron_variances


def log_likelihood(response_vector, position_idx, neuron_means, neuron_variances):
    """
    Compute Gaussian log-likelihood of observing response given position.

    Parameters
    ----------
    response_vector : array (n_neurons,) - observed response at one timepoint
    position_idx : int - index into angle_range (not the angle itself)
    neuron_means : array (n_neurons, n_positions)
    neuron_variances : array (n_neurons, n_positions)

    Returns
    -------
    ll : float - log-likelihood value
    """
    expected = neuron_means[:, position_idx]
    variances = neuron_variances[:, position_idx]

    # Add small epsilon to avoid division by zero
    variances = np.maximum(variances, 1e-6)

    # Gaussian log-likelihood (constant terms dropped since we only need argmax)
    ll = -np.sum((response_vector - expected)**2 / variances)
    return ll


def decode(response_vector, neuron_means, neuron_variances):
    """Find position that maximizes likelihood using empirical statistics."""
    n_positions = neuron_means.shape[1]
    ll_values = [log_likelihood(response_vector, p_idx, neuron_means, neuron_variances)
                 for p_idx in range(n_positions)]
    return device.angle_range[np.argmax(ll_values)]


# =============================================================================
# TRAIN/TEST UTILITIES
# =============================================================================

def get_train_test_idxs(train_pctg):
    """Split trials into train and test sets."""
    trial_idxs = device.get_condition_trials(monocular=None, distance=None)
    trial_idxs = np.random.permutation(trial_idxs)
    split_point = int(train_pctg * len(trial_idxs))

    train_idxs = trial_idxs[:split_point]
    test_idxs = trial_idxs[split_point:]

    return train_idxs, test_idxs


def get_test_trial_responses(cells_to_include, test_idx):
    """Get response vectors for all neurons for a single test trial."""
    test_responses = np.array([
        device.get_repeat_trace(cell, test_idx)
        for cell in cells_to_include
    ])
    return test_responses  # Shape: (n_neurons, n_positions)


def decode_trajectory(cells_to_include, neuron_means, neuron_variances, test_idx):
    """Decode position at each timepoint for a single test trial."""
    test_responses = get_test_trial_responses(cells_to_include, test_idx)

    decoded_positions = []
    for i in range(test_responses.shape[1]):  # For each position/timepoint
        decoded_positions.append(decode(test_responses[:, i], neuron_means, neuron_variances))

    return decoded_positions


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_decoder(actual, decoded):
    """Compute summary metrics for decoder performance."""
    errors = decoded - actual

    metrics = {
        'mae': np.mean(np.abs(errors)),
        'rmse': np.sqrt(np.mean(errors**2)),
        'bias': np.mean(errors),
        'correlation': np.corrcoef(actual.flatten(), decoded.flatten())[0, 1],
        'std': np.std(errors),
    }

    return metrics


def plot_decoder_performance(actual, decoded):
    """Comprehensive visualization of decoder performance."""
    errors = decoded - actual
    metrics = evaluate_decoder(actual, decoded)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Scatter (decoded vs actual)
    ax = axes[0, 0]
    ax.scatter(actual.flatten(), decoded.flatten(), alpha=0.3, s=5)
    lims = [min(actual.min(), decoded.min()), max(actual.max(), decoded.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual position (°)')
    ax.set_ylabel('Decoded position (°)')
    ax.set_title(f'Decoded vs Actual (r = {metrics["correlation"]:.3f})')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean decoded trajectory
    ax = axes[0, 1]
    mean_actual = np.mean(actual, axis=0)
    mean_decoded = np.mean(decoded, axis=0)
    sem_decoded = np.std(decoded, axis=0) / np.sqrt(decoded.shape[0])
    time_points = np.arange(len(mean_actual))

    ax.plot(time_points, mean_actual, 'k-', linewidth=2, label='Actual', alpha=0.7)
    ax.plot(time_points, mean_decoded, color='blue', linewidth=2, label='Decoded')
    ax.fill_between(time_points, mean_decoded - sem_decoded, mean_decoded + sem_decoded,
                    alpha=0.3, color='blue')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Position (°)')
    ax.set_title('Mean Trajectory (across trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error histogram
    ax = axes[1, 0]
    ax.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.axvline(metrics['bias'], color='orange', linestyle='-', linewidth=2,
               label=f'Bias = {metrics["bias"]:.2f}°')
    ax.set_xlabel('Error (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (MAE = {metrics["mae"]:.2f}°, RMSE = {metrics["rmse"]:.2f}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Error vs position
    ax = axes[1, 1]
    pos_flat = actual.flatten()
    err_flat = errors.flatten()
    bins = np.linspace(pos_flat.min(), pos_flat.max(), 15)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(pos_flat, bins) - 1

    mean_err = []
    std_err = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_err.append(np.mean(np.abs(err_flat[mask])))
            std_err.append(np.std(err_flat[mask]))
        else:
            mean_err.append(np.nan)
            std_err.append(np.nan)

    ax.errorbar(bin_centers, mean_err, yerr=std_err, fmt='o-', capsize=3)
    ax.axhline(metrics['mae'], color='r', linestyle='--', alpha=0.5,
               label=f'Overall MAE = {metrics["mae"]:.2f}°')
    ax.set_xlabel('Actual position (°)')
    ax.set_ylabel('Absolute error (°)')
    ax.set_title('Error vs Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, metrics


def print_metrics(metrics):
    """Pretty print the metrics."""
    print("=" * 40)
    print("DECODER PERFORMANCE")
    print("=" * 40)
    print(f"  Mean Absolute Error:  {metrics['mae']:.2f}°")
    print(f"  RMSE:                 {metrics['rmse']:.2f}°")
    print(f"  Bias:                 {metrics['bias']:.2f}°")
    print(f"  Correlation:          {metrics['correlation']:.3f}")
    print(f"  Error Std:            {metrics['std']:.2f}°")
    print("=" * 40)


# =============================================================================
# MAIN: RUN CROSS-VALIDATION
# =============================================================================

if __name__ == "__main__":

    # Identify reliable cells
    reliable_cells_df = viz.filter_by_reliability(session_df, reliable_type='reliable', threshold=0.85)
    reliable_cells = reliable_cells_df['cell_id'].values
    print(f'Using {len(reliable_cells)} reliable cells for decoding')

    # Perform k-fold cross-validation
    k_folds = 5
    all_decoded_positions = []
    all_test_idxs = []

    for fold in range(k_folds):
        print(f'\n=== Fold {fold + 1}/{k_folds} ===')

        # Get train/test split for this fold
        train_idxs, test_idxs = get_train_test_idxs(0.9)
        print(f'Training on {len(train_idxs)} trials, testing on {len(test_idxs)} trials')

        # Compute empirical tuning statistics on training data
        neuron_means, neuron_variances = compute_all_neuron_stats(reliable_cells, train_idxs)

        # Decode test trials
        fold_decoded_positions = []
        for i, test_idx in enumerate(test_idxs):
            print(f'Decoding trial {i + 1} of {len(test_idxs)}, index is {test_idx}')
            fold_decoded_positions.append(
                decode_trajectory(reliable_cells, neuron_means, neuron_variances, test_idx)
            )

        all_decoded_positions.extend(fold_decoded_positions)
        all_test_idxs.extend(test_idxs)

    print(f'\nTotal decoded trials: {len(all_decoded_positions)}')

    # Evaluate
    actual_positions = np.tile(device.angle_range, (len(all_decoded_positions), 1))
    decoded_positions_array = np.array(all_decoded_positions)

    print(f'Actual positions shape: {actual_positions.shape}')
    print(f'Decoded positions shape: {decoded_positions_array.shape}')

    fig, metrics = plot_decoder_performance(actual_positions, decoded_positions_array)
    print_metrics(metrics)
    plt.show()

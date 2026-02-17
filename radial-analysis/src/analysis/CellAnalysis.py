import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


class CellAnalyzer:
    """
    Analyzes individual cell responses from radial stimulus experiments.

    This class computes reliability metrics, modulation statistics, and ocular dominance
    for individual cells. It requires a RadialDevice instance to access the underlying data.

    Parameters
    ----------
    device : RadialDevice
        RadialDevice instance containing the loaded data and utility methods

    Examples
    --------
    >>> from src.RadialDevice import RadialDevice
    >>> from src.CellAnalysis import CellAnalyzer
    >>>
    >>> R = RadialDevice(mouse_name='SAM041', record_date='251128', experiment_name='tiling_3_120um')
    >>> analyzer = CellAnalyzer(R)
    >>> reliability = analyzer.analyze_cell_reliability(cell=10, monocular=True)
    >>> modulation = analyzer.analyze_cell_modulation(cell=10, monocular=True)
    """

    def __init__(self, device):
        """
        Initialize the analyzer with a RadialDevice instance.

        Parameters
        ----------
        device : RadialDevice
            RadialDevice instance with loaded data
        """
        self.device = device

    def compute_reliability(self, cell, distance=None, monocular=None, n_iterations=100):
        """
        Compute split-half reliability for a cell using correlation analysis.

        Randomly splits trials into two halves, computes mean response for each half,
        and calculates the correlation between them. Repeats this process n_iterations times.
        Also computes a null distribution by circularly shifting responses.

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        distance : int, optional
            Distance condition to filter trials (0-3, or 4 for control)
        monocular : bool, optional
            Whether to use monocular (True) or binocular (False) trials
        n_iterations : int, optional
            Number of split-half iterations (default: 100)

        Returns
        -------
        tuple of (corr, corr_shifted)
            corr : ndarray of shape (n_iterations,)
                Correlation coefficients for split-half comparisons
            corr_shifted : ndarray of shape (n_iterations,)
                Correlation coefficients for circularly-shifted null distribution
        """
        repeat_indices = self.device.get_condition_trials(monocular=monocular, distance=distance)
        idx = np.where(self.device.suite2p_id == cell)[0][0]

        all_traces = np.array([self.device.get_on_trace(idx, repeat).T for repeat in repeat_indices])

        n_repeats, n_time = all_traces.shape
        n_halves = int(np.ceil(n_repeats / 2))

        corr = np.zeros(n_iterations)
        corr_shifted = np.zeros(n_iterations)

        for i in range(n_iterations):
            # Fast shuffle and split
            shuffled_idx = np.random.permutation(n_repeats)
            first_idx = shuffled_idx[:n_halves]
            second_idx = shuffled_idx[n_halves:]

            # Mean traces
            first_mean = all_traces[first_idx].mean(axis=0)
            second_mean = all_traces[second_idx].mean(axis=0)
            corr[i] = np.corrcoef(first_mean, second_mean)[0, 1]

            # Shifted version
            shifts_first = np.random.randint(-180, 180, size=len(first_idx))
            shifts_second = np.random.randint(-180, 180, size=len(second_idx))

            time_idx = np.arange(n_time)
            first_shifted = all_traces[first_idx][
                np.arange(len(first_idx))[:, None],
                (time_idx - shifts_first[:, None]) % n_time
            ]
            second_shifted = all_traces[second_idx][
                np.arange(len(second_idx))[:, None],
                (time_idx - shifts_second[:, None]) % n_time
            ]

            corr_shifted[i] = np.corrcoef(
                first_shifted.mean(axis=0),
                second_shifted.mean(axis=0)
            )[0, 1]

        return corr, corr_shifted

    def analyze_cell_reliability(self, cell, n_iterations=100, corr_threshold=None, distance=None, monocular=None):
        """
        Analyze whether a cell's response is reliable by comparing with control condition.

        A cell is considered reliable if:
        1. Split-half correlation exceeds 99th percentile of null distribution
        2. Response to stimulus distances is higher than control condition

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        n_iterations : int, optional
            Number of iterations for reliability calculation (default: 100)
        corr_threshold : float, optional
            Minimum correlation threshold (if provided)
        distance : int, optional
            Distance condition to filter trials
        monocular : bool, optional
            Whether to analyze monocular condition

        Returns
        -------
        dict
            Dictionary containing:
            - reliable : bool
                Overall reliability determination
            - stimulus_reliable : bool
                Whether stimulus response is reliable
            - control_reliable : bool
                Whether control condition shows reliability
            - control_higher : bool
                Whether control RF values are higher than stimulus
            - mean_corr : float
                Mean split-half correlation
        """
        corr, corr_shifted = self.compute_reliability(cell, distance=distance, monocular=monocular, n_iterations=n_iterations)
        stimulus_reliable = np.mean(corr) > np.percentile(corr_shifted, 99)

        corr_control, corr_shifted_control = self.compute_reliability(cell, distance=4, monocular=monocular, n_iterations=n_iterations)
        control_reliable = np.mean(corr_control) > np.percentile(corr_shifted_control, 99.9)

        rf_values_distances = [np.mean(self.device.find_rf_values(cell, distance=distance, monocular=monocular)) for distance in [0, 1, 2, 3]]
        rf_values_control = np.mean(self.device.find_rf_values(cell, distance=4, monocular=monocular))

        # make sure that rf_values_distances are higher than rf_values_control, at least for three distances
        control_comparison = [rf_values_control > rf_values for rf_values in rf_values_distances]
        control_higher = np.sum(control_comparison) > 1

        if corr_threshold is not None:
            reliable = stimulus_reliable and not (control_higher) and np.mean(corr) > corr_threshold
        else:
            reliable = stimulus_reliable and not (control_higher)

        results = {
            'reliable': reliable,
            'stimulus_reliable': stimulus_reliable,
            'control_reliable': control_reliable,
            'control_higher': control_higher,
            'mean_corr': np.mean(corr)
        }

        return results

    def bootstrap_modulation(self, cell, monocular=None, window_size=5, n_iterations=200):
        """
        Analyze modulation strength using bootstrap resampling.

        For each bootstrap iteration, randomly splits trials into two halves:
        - First half: determines preferred and least-preferred distances
        - Second half: computes modulation index for those distances

        A cell is considered modulated if >95% of bootstrap iterations show positive modulation.

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        monocular : bool, optional
            Whether to analyze monocular condition
        window_size : int, optional
            Size of window around RF peak in frames (default: 5)
        n_iterations : int, optional
            Number of bootstrap iterations (default: 200)

        Returns
        -------
        dict
            Dictionary containing:
            - rf_position : float
                Peak receptive field position in degrees
            - preferred_distance : list
                Preferred distance for each bootstrap iteration
            - least_preferred_distance : list
                Least preferred distance for each bootstrap iteration
            - modulation_index : list
                Modulation indices from each iteration
            - bootstrap_p_value : float
                p-value for modulation (1 - fraction above zero)
            - preferred_distance_consistency : float
                Percentage consistency of preferred distance
            - bootstrap_modulated : bool
                Whether cell is significantly modulated (>95% iterations positive)
        """
        # Get mean trace and find RF peak window
        mean_trace = self.device.get_all_traces(cell, distance=None, monocular=monocular, mean_trace=True)[0]
        rf_idx = np.argmax(mean_trace)
        window_slice = slice(max(0, rf_idx - window_size), min(len(mean_trace), rf_idx + window_size + 1))

        # Get number of repeats per distance and validate
        n_repeats_per_distance = [
            self.device.get_all_traces(cell, distance=d, monocular=monocular, mean_trace=False).shape[0]
            for d in range(4)
        ]

        if len(set(n_repeats_per_distance)) != 1:
            print(f"Warning: Cell {cell} has different numbers of repeats per distance: {n_repeats_per_distance}")
            print("Using the minimum number of repeats across all distances.")

        n_repeats = min(n_repeats_per_distance)
        n_halves = (n_repeats + 1) // 2  # Equivalent to ceil(n_repeats / 2)

        # Pre-compute windowed mean traces for all distances: shape (4, n_repeats)
        trace_values = np.array([
            self.device.get_all_traces(cell, distance=d, monocular=monocular, mean_trace=False)[:n_repeats, window_slice].mean(axis=1)
            for d in range(4)
        ])

        # Pre-generate all random permutations at once: shape (n_iterations, n_repeats)
        halves_matrix = np.array([
            np.random.permutation(np.concatenate([np.ones(n_halves, dtype=bool), np.zeros(n_halves, dtype=bool)]))[:n_repeats]
            for _ in range(n_iterations)
        ])

        # Vectorized computation of mean values for both halves across all iterations
        # trace_values: (4, n_repeats), halves_matrix: (n_iterations, n_repeats)
        # Result shapes: (n_iterations, 4)
        mean_first_half = np.array([
            trace_values[:, mask].mean(axis=1) for mask in halves_matrix
        ])
        mean_second_half = np.array([
            trace_values[:, ~mask].mean(axis=1) for mask in halves_matrix
        ])

        # Find preferred/least preferred distances from first half
        preferred_distance = mean_first_half.argmax(axis=1)
        least_preferred_distance = mean_first_half.argmin(axis=1)

        # Calculate modulation index using second half values
        # Index into second half using the distances determined from first half
        iter_indices = np.arange(n_iterations)
        pref_values = np.abs(mean_second_half[iter_indices, preferred_distance])
        least_values = np.abs(mean_second_half[iter_indices, least_preferred_distance])

        modulation_indices = (pref_values - least_values) / (pref_values + least_values + 1e-10)

        unique_vals, counts = np.unique(preferred_distance, return_counts=True)
        preferred_distance_consistency = (np.max(counts) / len(preferred_distance)) * 100

        # Calculate statistics
        percentage_above_zero = (modulation_indices > 0).mean() * 100

        return {
            'rf_position': self.device.angle_range[rf_idx],
            'preferred_distance': preferred_distance.tolist(),
            'least_preferred_distance': least_preferred_distance.tolist(),
            'modulation_index': modulation_indices.tolist(),
            'bootstrap_p_value': 1 - (percentage_above_zero / 100),
            'preferred_distance_consistency': preferred_distance_consistency,
            'bootstrap_modulated': percentage_above_zero >= 95
        }

    def analyze_cell_modulation(self, cell, monocular=None, n_iterations=200):
        """
        Analyze cell modulation using both bootstrap and ANOVA methods.

        Combines bootstrap analysis with one-way ANOVA across distance conditions.

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        monocular : bool, optional
            Whether to analyze monocular condition
        n_iterations : int, optional
            Number of bootstrap iterations (default: 200)

        Returns
        -------
        dict
            Dictionary containing all bootstrap results plus:
            - anova_p_value : float
                p-value from one-way ANOVA across distances
            - anova_modulated : bool
                Whether cell is modulated by ANOVA (p < 0.05)
        """
        results = self.bootstrap_modulation(cell, monocular=monocular, n_iterations=n_iterations)

        # calculate the f_statistic and p_value
        rf_values = [self.device.find_rf_values(cell, distance=distance, monocular=monocular) for distance in [0, 1, 2, 3, 4]]
        distances_rf = [rf_values[i] for i in range(len(rf_values)-1)]
        f_stat, p_value = f_oneway(*distances_rf)
        results['anova_p_value'] = p_value
        results['anova_modulated'] = p_value < 0.05

        return results

    def analyze_ocular_dominance(self, cell):
        """
        Calculate ocular dominance index (ODI) for a cell.

        ODI measures the relative preference for binocular vs monocular stimulation
        at the cell's receptive field location.

        ODI = (binocular_response - monocular_response) / (|binocular| + |monocular|)

        ODI > 0 indicates binocular preference
        ODI < 0 indicates monocular preference

        Parameters
        ----------
        cell : int
            Cell ID to analyze

        Returns
        -------
        float
            Ocular dominance index between -1 and 1
        """
        monocular_mean = self.device.get_all_traces(cell, distance=None, monocular=True, mean_trace=True)[0]
        binocular_mean = self.device.get_all_traces(cell, distance=None, monocular=False, mean_trace=True)[0]

        receptive_field_position = self.device.find_rf_position(cell, distance=None, monocular=None)
        receptive_field_position_index = np.where(self.device.angle_range == receptive_field_position)[0][0]

        window_size = 10
        window_start = max(0, receptive_field_position_index - window_size)
        window_end = min(len(monocular_mean), receptive_field_position_index + window_size + 1)
        monocular_mean_rf = np.mean(monocular_mean[window_start:window_end])
        binocular_mean_rf = np.mean(binocular_mean[window_start:window_end])

        ocular_dominance_index = (binocular_mean_rf - monocular_mean_rf) / (np.abs(binocular_mean_rf) + np.abs(monocular_mean_rf) + 1e-10)

        return ocular_dominance_index

    def analyze_cell_stats(self, cell, n_iterations=200):
        """
        Compute comprehensive statistics for a single cell.

        Combines all analysis methods to produce a complete characterization including:
        - Receptive field location
        - Ocular dominance
        - Preferred distance
        - Reliability metrics for monocular and binocular conditions
        - Modulation metrics for monocular and binocular conditions

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        n_iterations : int, optional
            Number of bootstrap/reliability iterations (default: 200)

        Returns
        -------
        dict
            Nested dictionary containing:
            - cell_id : int
            - ODI : float (ocular dominance index)
            - RF : float (receptive field position in degrees)
            - distance : int (preferred distance across all conditions)
            - monocular : dict (reliability and modulation results)
            - binocular : dict (reliability and modulation results)
        """
        rf = self.device.find_rf_position(cell, distance=None, monocular=None)
        ODI = self.analyze_ocular_dominance(cell)

        rf_values = [np.mean(self.device.find_rf_values(cell, distance=i)) for i in range(4)]
        all_distance = np.argmax(rf_values)

        results = {
            'cell_id': cell,
            'session_name': self.device.session_name,
            'ODI': ODI.tolist(),
            'RF': rf.tolist(),
            'distance': all_distance.tolist(),
            'monocular': {},
            'binocular': {},
        }

        for condition, monocular in [('monocular', True), ('binocular', False)]:

            modulation_results = self.analyze_cell_modulation(cell, monocular=monocular)
            reliable_results = self.analyze_cell_reliability(cell, monocular=monocular, n_iterations=n_iterations)

            results[condition] = {
                'RF': modulation_results['rf_position'][0].tolist(),
                'reliable': reliable_results['reliable'],
                'reliable_corr': reliable_results['mean_corr'].tolist(),
                'bootstrap_pvalue': modulation_results['bootstrap_p_value'].tolist(),
                'anova_pvalue': modulation_results['anova_p_value'].tolist(),
                'consistency': modulation_results['preferred_distance_consistency'].tolist(),
                'DTI': np.mean(modulation_results['modulation_index']).tolist(),
                'distance': np.mean(modulation_results['preferred_distance']).tolist(),
            }

        return results

class SessionAnalyzer:
    """
    Analyzes all cells in a session and aggregates results.

    Parameters
    ----------
    device : RadialDevice
        RadialDevice instance containing the session data

    Examples
    --------
    >>> from src.RadialDevice import RadialDevice
    >>> from src.CellAnalysis import SessionAnalyzer
    >>>
    >>> R = RadialDevice(mouse_name='SAM041', record_date='251128', experiment_name='tiling_3_120um')
    >>> session_analyzer = SessionAnalyzer(R)
    >>> session_df = session_analyzer.analyze_session_stats()
    >>> # session_df is now a DataFrame with one row per cell
    """

    def __init__(self, device):
        """
        Initialize the session analyzer.

        Parameters
        ----------
        device : RadialDevice
            RadialDevice instance with loaded data
        """
        self.device = device
        self.cell_analyzer = CellAnalyzer(device)

    def analyze_session_stats(self, n_iterations=200):
        """
        Analyze all cells in the session and return a summary DataFrame.

        Parameters
        ----------
        n_iterations : int, optional
            Number of bootstrap/reliability iterations per cell (default: 200)

        Returns
        -------
        pd.DataFrame
            DataFrame where each row represents one cell, with columns:
            - cell_id : Cell identifier
            - ODI : Ocular dominance index
            - RF : Receptive field position
            - distance : Preferred distance
            - monocular : Dict of monocular condition statistics
            - binocular : Dict of binocular condition statistics
        """
        # Initialize empty lists to collect data for each cell
        cell_data = []

        # Analyze each cell
        for cell in self.device.suite2p_id:

            print(f'Analyzing cell {cell} of {self.device.suite2p_id[-1]}', end='\r', flush=True)

            results = self.cell_analyzer.analyze_cell_stats(cell, n_iterations=n_iterations)
            cell_data.append(results)

        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)

        return stats_df

    def save_dataframe(self, df, save_path=None):
        """
        Save a DataFrame to disk.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        save_path : str, optional
            Path to save the DataFrame. If None, saves to device.data_dir
            with automatic naming. Supports .csv and .pkl extensions.

        Returns
        -------
        str
            Full path where the DataFrame was saved
        """
        if save_path is None:
            save_path = f'{self.device.session_name}_dataframe.pkl'

        full_save_path = os.path.join(self.device.data_dir, save_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        if full_save_path.endswith('.csv'):
            df.to_csv(full_save_path, index=False)
        else:
            df.to_pickle(full_save_path)

        print(f'Saved DataFrame to: {full_save_path}')

        return full_save_path

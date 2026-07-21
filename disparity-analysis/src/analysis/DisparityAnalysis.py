import os
import numpy as np
import pandas as pd

from src.analysis.CellAnalysis import split_half_reliability


class DisparityCellAnalyzer:
    """
    Per-cell analysis for a binocular disparity session.

    Computes split-half reliability and the peak (RF) response per disparity,
    reusing :func:`split_half_reliability`. Requires a ``DisparityDevice``.

    Examples
    --------
    >>> from src.data.DisparityDevice import DisparityDevice
    >>> from src.analysis.DisparityAnalysis import DisparityCellAnalyzer
    >>> D = DisparityDevice(mouse_name='SAM045', record_date='260713',
    ...                         experiment_name='goggles-disparity')
    >>> DisparityCellAnalyzer(D).analyze_cell_reliability(cell=8, disparity=0)
    """

    def __init__(self, device):
        self.device = device

    def compute_reliability(self, cell, disparity, n_iterations=100, shift_window=None):
        """Split-half reliability of a cell's response at one disparity."""
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        repeat_indices = self.device.get_disparity_repeats(disparity)

        traces = [self.device.get_on_trace(idx, repeat).T for repeat in repeat_indices]
        if len(traces) == 0:
            return np.full(n_iterations, np.nan), np.full(n_iterations, np.nan)

        min_len = min(t.shape[0] for t in traces)
        all_traces = np.array([t[:min_len] for t in traces], dtype=float)

        return split_half_reliability(all_traces, n_iterations=n_iterations, shift_window=shift_window)

    def analyze_cell_reliability(self, cell, disparity, n_iterations=200, stimulus_percentile=99):
        """
        Whether a cell is reliably driven at one disparity.

        Reliable when the mean split-half correlation exceeds the
        ``stimulus_percentile`` of the shifted-null distribution.

        Returns
        -------
        dict
            {'reliable': bool, 'mean_corr': float}
        """
        corr, corr_shifted = self.compute_reliability(
            cell=cell, disparity=disparity, n_iterations=n_iterations
        )
        mean_corr = np.nanmean(corr)
        reliable = mean_corr > np.nanpercentile(corr_shifted, stimulus_percentile)
        return {'reliable': bool(reliable), 'mean_corr': float(mean_corr)}

    def analyze_cell_stats(self, cell, n_iterations=200):
        """
        Reliability + RF summary for one cell, as a tidy row per disparity.

        Returns
        -------
        list of dict
            One row per disparity with keys ``cell``, ``session_name``,
            ``disparity``, ``reliable``, ``mean_corr``, ``rf`` (peak frame),
            ``rf_value`` (mean response there), and ``rf_deg`` (peak angle).
        """
        cell = int(cell)
        angle_range = self.device.get_angle_range()

        rows = []
        for disparity in self.device.disparities:
            rel = self.analyze_cell_reliability(cell, disparity=disparity, n_iterations=n_iterations)
            rf_idx, rf_val = self.device.find_rf_position(cell, disparity=disparity)
            rows.append({
                'cell': cell,
                'session_name': self.device.session_name,
                'disparity': int(disparity),
                'reliable': rel['reliable'],
                'mean_corr': rel['mean_corr'],
                'rf': rf_idx,
                'rf_value': rf_val,
                'rf_deg': float(angle_range[rf_idx]),
            })
        return rows


class DisparitySessionAnalyzer:
    """
    Aggregate per-cell disparity stats across a whole session.

    Produces a **tidy** (long) DataFrame: one row per (cell, disparity).

    Examples
    --------
    >>> from src.analysis.DisparityAnalysis import DisparitySessionAnalyzer
    >>> df = DisparitySessionAnalyzer(D).analyze_session_stats()   # cell x disparity
    """

    def __init__(self, device):
        self.device = device
        self.cell_analyzer = DisparityCellAnalyzer(device)

    def analyze_session_stats(self, n_iterations=200):
        """Run :meth:`DisparityCellAnalyzer.analyze_cell_stats` for every cell."""
        rows = []
        for cell in self.device.suite2p_id:
            print(f'Analyzing cell {cell} of {self.device.suite2p_id[-1]}', end='\r', flush=True)
            rows.extend(self.cell_analyzer.analyze_cell_stats(int(cell), n_iterations=n_iterations))

        return pd.DataFrame(rows)

    def save_dataframe(self, df, save_path=None):
        """Save a stats DataFrame next to the session data (.pkl or .csv)."""
        if save_path is None:
            save_path = f'{self.device.session_name}_dataframe.pkl'

        full_save_path = os.path.join(self.device.data_dir, save_path)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        if full_save_path.endswith('.csv'):
            df.to_csv(full_save_path, index=False)
        else:
            df.to_pickle(full_save_path)

        print(f'Saved DataFrame to: {full_save_path}')
        return full_save_path

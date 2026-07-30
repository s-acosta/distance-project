import os
import numpy as np
import pandas as pd

from src.analysis.CellAnalysis import split_half_reliability


class GratingCellAnalyzer:
    """
    Per-cell analysis for a binocular phase-disparity grating session.

    Computes, per disparity, split-half reliability (reusing
    :func:`split_half_reliability`) and a full-field response magnitude. A grating
    is full-field with no spatial preference, so the response is summarised as the
    mean of the stim-on trace over the whole on-window rather than a peak-frame
    (RF) window. Requires a :class:`GratingDevice`.

    Examples
    --------
    >>> from src.data.GratingDevice import GratingDevice
    >>> from src.analysis.GratingAnalysis import GratingCellAnalyzer
    >>> G = GratingDevice(mouse_name='SAM045', record_date='260721',
    ...                   experiment_name='goggles-gratings')
    >>> GratingCellAnalyzer(G).analyze_cell_reliability(cell=8, disparity=0)
    """

    def __init__(self, device):
        self.device = device

    def _disparity_traces(self, cell, disparity):
        """
        ``(n_trials, min_len)`` stim-on trace stack for one disparity.

        Traces are truncated to the shortest trial before stacking (ragged-safe),
        the same pattern the reliability code uses. Built once and reused for both
        reliability and the response magnitude.
        """
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        repeat_indices = self.device.get_disparity_repeats(disparity)

        traces = [self.device.get_on_trace(idx, repeat).T for repeat in repeat_indices]
        if len(traces) == 0:
            return np.empty((0, 0))
        min_len = min(t.shape[0] for t in traces)
        return np.array([t[:min_len] for t in traces], dtype=float)

    def analyze_cell_reliability(self, cell, disparity, n_iterations=200,
                                 stimulus_percentile=99, traces=None):
        """
        Whether a cell is reliably driven at one disparity.

        Reliable when the mean split-half correlation exceeds the
        ``stimulus_percentile`` of the shifted-null distribution.

        Returns
        -------
        dict
            {'reliable': bool, 'mean_corr': float}
        """
        if traces is None:
            traces = self._disparity_traces(cell, disparity)
        corr, corr_shifted = split_half_reliability(traces, n_iterations=n_iterations)
        mean_corr = np.nanmean(corr)
        reliable = mean_corr > np.nanpercentile(corr_shifted, stimulus_percentile)
        return {'reliable': bool(reliable), 'mean_corr': float(mean_corr)}

    def analyze_cell_stats(self, cell, n_iterations=200):
        """
        Reliability + response magnitude per disparity, as a tidy row per disparity.

        Returns
        -------
        list of dict
            One row per disparity with keys ``cell``, ``session_name``,
            ``disparity``, ``reliable``, ``mean_corr``, ``response_mean``,
            ``response_sem`` (mean/sem across trials of the full-window mean
            response), and ``n_trials``.
        """
        cell = int(cell)
        rows = []
        for disparity in self.device.disparities:
            traces = self._disparity_traces(cell, disparity)
            rel = self.analyze_cell_reliability(
                cell, disparity, n_iterations=n_iterations, traces=traces
            )

            if traces.shape[0] > 0 and traces.shape[1] > 0:
                per_trial = traces.mean(axis=1)      # full-window mean response per trial
                response_mean = float(np.mean(per_trial))
                response_sem = float(np.std(per_trial) / np.sqrt(len(per_trial)))
                n_trials = int(traces.shape[0])
            else:
                response_mean = np.nan
                response_sem = np.nan
                n_trials = 0

            rows.append({
                'cell': cell,
                'session_name': self.device.session_name,
                'disparity': int(disparity),
                'reliable': rel['reliable'],
                'mean_corr': rel['mean_corr'],
                'response_mean': response_mean,
                'response_sem': response_sem,
                'n_trials': n_trials,
            })
        return rows


class GratingSessionAnalyzer:
    """
    Aggregate per-cell grating stats across a whole session.

    Produces a **tidy** (long) DataFrame: one row per (cell, disparity). The
    reliability columns (``cell``, ``disparity``, ``reliable``, ``mean_corr``)
    are a superset of the disparity session's, so the df is compatible with
    ``CrossExperimentAnalysis`` if this experiment is ever added to the cross-
    experiment comparison.

    Examples
    --------
    >>> from src.analysis.GratingAnalysis import GratingSessionAnalyzer
    >>> df = GratingSessionAnalyzer(G).analyze_session_stats()   # cell x disparity
    """

    def __init__(self, device):
        self.device = device
        self.cell_analyzer = GratingCellAnalyzer(device)

    def analyze_session_stats(self, n_iterations=200):
        """Run :meth:`GratingCellAnalyzer.analyze_cell_stats` for every cell."""
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

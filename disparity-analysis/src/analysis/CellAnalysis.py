import numpy as np
import pandas as pd
import os


def split_half_reliability(all_traces, n_iterations=100, shift_window=None):
    """
    Split-half reliability from a stack of single-trial traces.

    Randomly splits the trials in half, correlates the two mean traces, and
    repeats ``n_iterations`` times. A circularly-shifted null is computed in
    parallel. Paradigm-independent: the caller decides which trials go in
    (an eye's trials, a disparity's trials, ...).

    Parameters
    ----------
    all_traces : ndarray, shape (n_repeats, n_time)
        Single-trial traces, already truncated to a common length.
    n_iterations : int
    shift_window : int or None
        Max circular shift for the null; defaults to half the trace length.

    Returns
    -------
    (corr, corr_shifted) : tuple of ndarray
        Split-half correlations and shifted-null correlations.
    """
    all_traces = np.asarray(all_traces, dtype=float)
    if all_traces.ndim != 2 or all_traces.shape[0] < 2:
        return np.full(n_iterations, np.nan), np.full(n_iterations, np.nan)

    n_repeats, n_time = all_traces.shape
    n_halves = int(np.ceil(n_repeats / 2))
    max_shift = n_time // 2 if shift_window is None else int(min(abs(shift_window), n_time))

    corr = np.zeros(n_iterations, dtype=float)
    corr_shifted = np.zeros(n_iterations, dtype=float)

    def _safe_corr(a, b):
        c = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(c) else c

    time_idx = np.arange(n_time)

    for i in range(n_iterations):
        shuffled_idx = np.random.permutation(n_repeats)
        first_idx = shuffled_idx[:n_halves]
        second_idx = shuffled_idx[n_halves:]

        first_mean = all_traces[first_idx].mean(axis=0)
        second_mean = all_traces[second_idx].mean(axis=0)
        corr[i] = _safe_corr(first_mean, second_mean)

        shifts_first = np.random.randint(-max_shift, max_shift + 1, size=len(first_idx))
        shifts_second = np.random.randint(-max_shift, max_shift + 1, size=len(second_idx))

        first_shifted = all_traces[first_idx][
            np.arange(len(first_idx))[:, None],
            (time_idx - shifts_first[:, None]) % n_time,
        ]
        second_shifted = all_traces[second_idx][
            np.arange(len(second_idx))[:, None],
            (time_idx - shifts_second[:, None]) % n_time,
        ]

        corr_shifted[i] = _safe_corr(first_shifted.mean(axis=0), second_shifted.mean(axis=0))

    return corr, corr_shifted


class CellAnalyzer:
    """
    Per-cell analysis for a disparity (left/right eye) session.

    Computes split-half reliability per eye, eye-dominance (ODI), and the peak
    receptive-field response. Requires a ``MonocularDevice`` for data access.

    Examples
    --------
    >>> from src.data.MonocularDevice import MonocularDevice
    >>> from src.analysis.CellAnalysis import CellAnalyzer
    >>> D = MonocularDevice(mouse_name='SAM045', record_date='260609', experiment_name='goggles_left')
    >>> analyzer = CellAnalyzer(D)
    >>> analyzer.analyze_cell_reliability(cell=116, eye='left')
    """

    def __init__(self, device):
        self.device = device

    def compute_reliability(self, cell, eye=None, n_iterations=100, shift_window=None, **cond):
        """
        Split-half reliability of a cell's response for an eye (+ any other
        ``**cond``, e.g. ``elevation=120``) condition.

        Randomly splits the trials in half, correlates the two mean traces, and
        repeats ``n_iterations`` times. A circularly-shifted null is computed in
        parallel. Trace lengths are truncated to the shortest trial. Trial
        selection is delegated to the device's :meth:`_get_repeats` hook (same
        as :meth:`GogglesDevice.get_all_traces`), so any condition it supports
        works here too.

        Returns
        -------
        (corr, corr_shifted) : tuple of ndarray
            Split-half correlations and shifted-null correlations.
        """
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        repeat_indices = self.device._get_repeats(eye=eye, **cond)

        traces = [self.device.get_on_trace(idx, repeat).T for repeat in repeat_indices]
        if len(traces) == 0:
            return np.full(n_iterations, np.nan), np.full(n_iterations, np.nan)

        min_len = min(t.shape[0] for t in traces)
        all_traces = np.array([t[:min_len] for t in traces], dtype=float)

        return split_half_reliability(all_traces, n_iterations=n_iterations, shift_window=shift_window)

    def analyze_cell_reliability(self, cell, eye, n_iterations=200, stimulus_percentile=99, **cond):
        """
        Whether a cell is reliably driven by the stimulus in one eye (+ any
        other ``**cond``, e.g. ``elevation=120``).

        Reliable when the mean split-half correlation exceeds the
        ``stimulus_percentile`` of the shifted-null distribution.

        Returns
        -------
        dict
            {'reliable': bool, 'mean_corr': float}
        """
        corr, corr_shifted = self.compute_reliability(
            cell=cell, eye=eye, n_iterations=n_iterations, **cond
        )

        mean_corr = np.nanmean(corr)
        reliable = mean_corr > np.nanpercentile(corr_shifted, stimulus_percentile)

        return {
            'reliable': bool(reliable),
            'mean_corr': float(mean_corr),
        }

    def analyze_ocular_dominance(self, cell, **cond):
        """
        Ocular dominance index from the per-eye RF peak responses (+ any other
        ``**cond``, e.g. ``elevation=120``).

        ODI = (|right| - |left|) / (|right| + |left|);  >0 right-eye preference,
        <0 left-eye preference. Returns None for single-eye sessions.
        """
        if self.device.single_eye:
            return None

        _, right_val = self.device.find_rf_position(cell, eye='right', **cond)
        _, left_val = self.device.find_rf_position(cell, eye='left', **cond)

        denom = np.abs(right_val) + np.abs(left_val)
        if denom == 0:
            return None
        return float((np.abs(right_val) - np.abs(left_val)) / denom)

    def analyze_cell_stats(self, cell, n_iterations=200, **cond):
        """
        Reliability + RF summary for one cell (+ any other ``**cond``, e.g.
        ``elevation=120`` on a session with ``device.has_elevation``).

        For a single-eye session the columns are un-prefixed (``reliable``,
        ``mean_corr``, ``rf``, ``rf_value``, ``px``). For a both-eye session the
        columns are prefixed per eye (``right_*`` / ``left_*``) and an ``ODI`` is
        included. ``px`` uses signed screen coordinates (left negative) only when
        both eyes are present, matching the notebook. Any ``**cond`` passed in
        is stamped onto the row as-is (e.g. an ``elevation`` column), so rows
        from different conditions concatenate into one tidy DataFrame.
        """
        cell = int(cell)
        signed = not self.device.single_eye

        row = {'cell': cell, 'session_name': self.device.session_name}

        for eye in self.device.present_eyes:
            rel = self.analyze_cell_reliability(cell, eye=eye, n_iterations=n_iterations, **cond)
            rf_idx, rf_val = self.device.find_rf_position(cell, eye=eye, **cond)
            px = self.device.get_screen_range(eye, signed=signed)[rf_idx]

            prefix = '' if self.device.single_eye else f'{eye}_'
            row[f'{prefix}reliable'] = rel['reliable']
            row[f'{prefix}mean_corr'] = rel['mean_corr']
            row[f'{prefix}rf'] = rf_idx
            row[f'{prefix}rf_value'] = rf_val
            row[f'{prefix}px'] = px

        if not self.device.single_eye:
            row['ODI'] = self.analyze_ocular_dominance(cell, **cond)

        row.update(cond)
        return row


class SessionAnalyzer:
    """
    Aggregate per-cell stats across a whole disparity session.

    Examples
    --------
    >>> from src.analysis.CellAnalysis import SessionAnalyzer
    >>> session = SessionAnalyzer(D)
    >>> df = session.analyze_session_stats()   # one row per cell
    """

    def __init__(self, device):
        self.device = device
        self.cell_analyzer = CellAnalyzer(device)

    def analyze_session_stats(self, n_iterations=200, **cond):
        """
        Run :meth:`CellAnalyzer.analyze_cell_stats` for every cell.

        With no ``**cond``, pools every trial regardless of condition -- for a
        plain (non-elevation) session that's everything, unchanged from before.
        For a session with ``device.has_elevation``, that pools *across*
        elevations, which is rarely what you want; pass ``elevation=120`` (or
        whatever your center/default elevation is) to restrict to one condition,
        the same way a non-elevation session's single sweep always behaved. See
        :meth:`analyze_by_elevation` to instead get every elevation at once.
        """
        rows = []
        for cell in self.device.suite2p_id:
            print(f'Analyzing cell {cell} of {self.device.suite2p_id[-1]}', end='\r', flush=True)
            rows.append(self.cell_analyzer.analyze_cell_stats(int(cell), n_iterations=n_iterations, **cond))

        return pd.DataFrame(rows)

    def analyze_by_elevation(self, n_iterations=200):
        """
        Run :meth:`analyze_session_stats` separately for every elevation this
        session swept, and concatenate the results into one long-format
        DataFrame (one block of per-cell rows per elevation, tagged by an
        ``elevation`` column). Requires ``device.has_elevation``.
        """
        if not self.device.has_elevation:
            raise ValueError(
                f"{self.device.session_name} has no elevation sweep -- there's only one "
                "condition, so there's nothing to break out by elevation. Use "
                "analyze_session_stats() instead."
            )
        dfs = []
        for elev in self.device.elevations:
            elev = float(elev)
            print(f'=== Elevation {elev:.0f} px ===' + ' ' * 20)
            df = self.analyze_session_stats(n_iterations=n_iterations, elevation=elev)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

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

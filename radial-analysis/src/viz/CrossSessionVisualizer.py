import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt


class CrossSessionVisualizer:
    """
    Stacks `CellVisualizer.plot_all_traces_colormap_distances`-style rows from
    multiple RadialDevice sessions (e.g. different stimulus directions) into one
    comparison figure per cell.

    Assumes `cell` is a single ID valid across every session passed in (true when
    sessions share a `cell_id` space, e.g. split from the same suite2p run).
    """

    def __init__(self, session_vizs, stats=None):
        """
        Parameters
        ----------
        session_vizs : list of (str, CellVisualizer)
            Ordered list of (label, CellVisualizer) pairs, one per session to stack.
            Stacking order follows list order; within each session, a monocular row
            is placed before a binocular row (whichever the session has available).
            Example:
                [('temporal', CellVisualizer(device_temporal)),
                 ('nasal',    CellVisualizer(device_nasal))]
        stats : dict of str -> pd.DataFrame, optional
            Session DataFrames (from `analyze_session` / `SessionAnalyzer`), keyed by
            the same labels. If given, each cell's stats are read from these instead
            of re-running `analyze_cell_stats`: much faster, and the figure then agrees
            with the saved tables (the reliability and bootstrap tests are random).
        """
        self.session_vizs = list(session_vizs)
        self.stats = None if stats is None else {
            label: df.set_index('cell_id') for label, df in stats.items()
        }

    def _build_rows(self, conditions=None):
        """Returns a list of (label, viz, condition) tuples, one per row to draw.

        Parameters
        ----------
        conditions : iterable of str, optional
            Subset of {'monocular', 'binocular'} to include as rows. If None
            (default), every condition available in each session is included.
        """
        rows = []
        for label, viz in self.session_vizs:
            has_monocular, has_binocular = viz.analyzer._get_available_conditions()
            if has_monocular and (conditions is None or 'monocular' in conditions):
                rows.append((label, viz, {'monocular': True, 'title': 'monocular'}))
            if has_binocular and (conditions is None or 'binocular' in conditions):
                rows.append((label, viz, {'monocular': False, 'title': 'binocular'}))
        return rows

    def plot_cell(self, cell, shared_scale=True, conditions=None, show_condition=True):
        """
        Create a stacked comparison figure for one cell across all sessions.

        Parameters
        ----------
        cell : int
            Cell ID to plot (must be valid across all sessions passed to __init__).
        shared_scale : bool, optional
            If True (default), heatmap color and y-limits are computed once, pooled
            across every session, so panels are directly comparable across sessions.
            If False, each session gets its own scale (same formula as
            `CellVisualizer.plot_all_traces_colormap_distances`).
        conditions : iterable of str, optional
            Subset of {'monocular', 'binocular'} rows to draw, e.g. ['binocular']
            to show only the binocular row for every session. If None (default),
            every condition available in each session is drawn (previous behavior).
            "Reliable General" in the stats column still reflects the full session
            (both conditions), not just whichever rows are shown.
        show_condition : bool, optional
            If False, rows are titled by session label only (e.g. left / both /
            right-eye sessions that are all flagged binocular). Default True.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """

        rows = self._build_rows(conditions)
        n_rows = len(rows)

        fig, axs = plt.subplots(n_rows, 5, figsize=(22, n_rows * 4),
                               gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.2]},
                               squeeze=False)
        for r in range(n_rows):
            axs[r, 1].remove()
        plt.subplots_adjust(wspace=0.4, hspace=0.3)

        # Per-session (not per-row) stats and scale, reused across that session's rows
        session_results = {}
        session_reliable = {}
        session_vmin_vmax = {}
        for label, viz in self.session_vizs:
            has_monocular, has_binocular = viz.analyzer._get_available_conditions()
            conditions = []
            if has_monocular:
                conditions.append({'monocular': True, 'title': 'monocular'})
            if has_binocular:
                conditions.append({'monocular': False, 'title': 'binocular'})

            if self.stats is not None:
                results = self.stats[label].loc[cell].to_dict()
            else:
                results = viz.analyzer.analyze_cell_stats(cell)
            session_results[label] = results
            session_reliable[label] = any(results[c['title']]['reliable'] for c in conditions)

            all_traces = viz.device.get_all_traces(cell, distance=None, monocular=None, mean_trace=False)
            session_vmin_vmax[label] = np.percentile(all_traces, [1, 99])

        if shared_scale:
            pooled = np.concatenate([
                viz.device.get_all_traces(cell, distance=None, monocular=None, mean_trace=False).ravel()
                for _, viz in self.session_vizs
            ])
            vmin, vmax = np.percentile(pooled, [1, 99])

        for row_idx, (label, viz, condition) in enumerate(rows):
            row_vmin, row_vmax = (vmin, vmax) if shared_scale else session_vmin_vmax[label]
            viz._draw_condition_row(
                fig, axs, row_idx, cell, condition,
                session_results[label], session_reliable[label],
                row_vmin, row_vmax, row_label=label, show_condition=show_condition,
            )

        labels_str = ' vs '.join(label for label, _ in self.session_vizs)
        fig.suptitle(f'Cell {cell} — {labels_str}', fontsize=20)
        plt.tight_layout()

        return fig

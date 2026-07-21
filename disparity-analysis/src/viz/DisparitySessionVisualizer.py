import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import setup_plot_style
from src.analysis.DisparityAnalysis import DisparitySessionAnalyzer
from src.viz._base import distance_palette

setup_plot_style()


class DisparitySessionVisualizer:
    """
    Session-level summaries for a binocular disparity session.

    Prints per-disparity reliable-cell counts and draws the population plots:
    per-disparity RF (in degrees) histograms and preferred-angle vs disparity.
    Works off the tidy (long) stats DataFrame from ``DisparitySessionAnalyzer``,
    computed once and cached; pass a precomputed ``stats_df`` to skip it.

    Examples
    --------
    >>> from src.data.DisparityDevice import DisparityDevice
    >>> from src.viz.DisparitySessionVisualizer import DisparitySessionVisualizer
    >>> D = DisparityDevice(mouse_name='SAM045', record_date='260713',
    ...                         experiment_name='goggles-disparity')
    >>> sv = DisparitySessionVisualizer(D)
    >>> sv.print_summary()
    >>> sv.plot_summary()
    """

    def __init__(self, device, stats_df=None, n_iterations=200):
        self.device = device
        self.analyzer = DisparitySessionAnalyzer(device)
        self.n_iterations = n_iterations
        self.stats_df = stats_df
        setup_plot_style()

    # ----------------------------------------------------------------- stats

    def compute_stats(self, force=False):
        """Return the tidy per-(cell, disparity) stats DataFrame, cached."""
        if self.stats_df is None or force:
            print(f"Computing session stats ({len(self.device.suite2p_id)} cells x "
                  f"{self.device.n_disparities} disparities, n_iterations={self.n_iterations})...")
            self.stats_df = self.analyzer.analyze_session_stats(n_iterations=self.n_iterations)
        return self.stats_df

    def _reliable_rows(self, df, corr_threshold=None):
        """Boolean mask of reliable rows, optionally with a ``mean_corr`` floor."""
        mask = df['reliable'].astype(bool)
        if corr_threshold is not None:
            mask &= df['mean_corr'] >= corr_threshold
        return mask

    def reliable_sets(self, corr_threshold=None):
        """
        {disparity: set(reliable cells)} for each disparity, plus 'any', 'none',
        'all'.

        When ``corr_threshold`` is given, a cell counts as reliable at a disparity
        only if it also has ``mean_corr >= corr_threshold`` there.
        """
        df = self.compute_stats()
        all_cells = set(int(c) for c in df['cell'].unique())
        reliable = self._reliable_rows(df, corr_threshold)

        sets = {}
        for d in self.device.disparities:
            rel = df[(df['disparity'] == d) & reliable]
            sets[d] = set(int(c) for c in rel['cell'])

        any_reliable = set().union(*sets.values()) if sets else set()
        sets['any'] = any_reliable
        sets['none'] = all_cells - any_reliable
        sets['all'] = all_cells
        return sets

    def _disparity_colors(self):
        """One distinct color per disparity, sampled along the distance colormap scale."""
        return distance_palette(self.device.n_disparities)

    def print_summary(self, corr_threshold=None):
        """Print reliable-cell counts per disparity and across any disparity.

        ``corr_threshold`` adds a ``mean_corr`` floor on top of the reliability test.
        """
        df = self.compute_stats()
        sets = self.reliable_sets(corr_threshold)
        n = len(df['cell'].unique())
        line = "=" * 50

        def pct(k):
            return f"{100 * k / n:5.1f}%" if n else "  n/a"

        print(line)
        print(f" Session: {self.device.session_name}")
        print(f" Disparities (deg): {self.device.disparities}")
        if corr_threshold is not None:
            print(f" mean_corr floor: {corr_threshold}")
        print(f" Total cells: {n}")
        print("-" * 50)
        for d in self.device.disparities:
            k = len(sets[d])
            print(f" {f'Disparity {d:+d}° reliable':26s}: {k:4d}  ({pct(k)})")
        print("-" * 50)
        print(f" {'Any disparity':26s}: {len(sets['any']):4d}  ({pct(len(sets['any']))})")
        print(f" {'None':26s}: {len(sets['none']):4d}  ({pct(len(sets['none']))})")
        print(line)
        return sets

    # ----------------------------------------------------------- population plots

    def plot_summary(self, print_counts=True, corr_threshold=None):
        """
        Per-disparity RF-in-degrees histograms (one purple shade each) plus a
        preferred-angle vs disparity panel. Returns the figure.

        ``corr_threshold`` adds a ``mean_corr`` floor on top of the reliability test.
        """
        if print_counts:
            self.print_summary(corr_threshold)

        df = self.compute_stats()
        reliable = self._reliable_rows(df, corr_threshold)
        disparities = self.device.disparities
        colors = self._disparity_colors()
        angle = self.device.get_angle_range()
        bins = np.linspace(angle[0], angle[-1], 25)

        n_panels = len(disparities) + 1
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        flat = axes.flat

        # per-disparity RF-in-degrees histograms (reliable cells only)
        means, sems, xs = [], [], []
        for i, (d, color) in enumerate(zip(disparities, colors)):
            ax = flat[i]
            rf_deg = df[(df['disparity'] == d) & reliable]['rf_deg'].dropna()
            ax.hist(rf_deg, bins=bins, color=color, edgecolor='black', alpha=0.9)
            ax.set_title(f'Disparity {d:+d}°  (n={len(rf_deg)})', fontsize=16)
            ax.set_xlabel('RF angle (deg)', fontsize=14)
            ax.set_ylabel('Number of cells', fontsize=14)
            if len(rf_deg):
                means.append(float(rf_deg.mean()))
                sems.append(float(rf_deg.std() / np.sqrt(len(rf_deg))))
                xs.append(d)

        # preferred-angle vs disparity
        ax = flat[len(disparities)]
        if xs:
            ax.errorbar(xs, means, yerr=sems, marker='o', color='indigo',
                        linewidth=2, capsize=4)
        ax.set_xlabel('Disparity (deg)', fontsize=14)
        ax.set_ylabel('Mean RF angle (deg)', fontsize=14)
        ax.set_title('Preferred angle vs disparity', fontsize=16)

        # hide any unused panels
        for j in range(n_panels, nrows * ncols):
            flat[j].axis('off')

        plt.tight_layout()
        return fig

"""
Cross-experiment reliability: cells reliable in BOTH the distance (radial) and
goggles (disparity) experiments.

Works purely on the two per-cell stats DataFrames that each experiment's
``SessionAnalyzer.analyze_session_stats()`` produces (and ``save_dataframe()``
persists). Reading DataFrames needs only pandas, so this never imports either
experiment's ``src`` package — no namespace collision.

Reliability uses each experiment's own definition, already baked into its df:
  - distance (radial): reliable in any available monocular/binocular condition
    (radial's split-half-vs-null + control-distance check).
  - goggles (disparity): reliable in either eye (or the single eye present).

Cell ids correspond only for a same-suite2p-run split pair; ``__init__`` warns
if the overlap is empty.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CrossExperimentAnalysis:

    def __init__(self, radial_df, disparity_df, how='inner'):
        self.radial_df = radial_df
        self.disparity_df = disparity_df

        rad = self._radial_reliability(radial_df)
        disp = self._disparity_reliability(disparity_df)

        self.merged = pd.merge(rad, disp, on='cell', how=how)
        self.merged['reliable_both'] = (
            self.merged['distance_reliable'] & self.merged['goggles_reliable']
        )

        if len(self.merged) == 0:
            print("[warning] no common cells between the two DataFrames — the sessions "
                  "may not be a same-suite2p-run split pair, so cell ids don't correspond.")

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_paths(cls, radial_path, disparity_path, **kwargs):
        """Load both per-cell stats DataFrames from pickle files."""
        return cls(pd.read_pickle(radial_path), pd.read_pickle(disparity_path), **kwargs)

    @classmethod
    def from_devices(cls, distance_device, disparity_device, **kwargs):
        """Load each device's default ``{session_name}_dataframe.pkl`` from its data_dir."""
        def _path(device):
            return os.path.join(device.data_dir, f"{device.session_name}_dataframe.pkl")
        return cls.from_paths(_path(distance_device), _path(disparity_device), **kwargs)

    # ------------------------------------------------------ reliability extract

    @staticmethod
    def _cond_reliable(cond):
        """Reliability flag from a radial monocular/binocular condition dict."""
        return isinstance(cond, dict) and bool(cond.get('reliable', False))

    @staticmethod
    def _cond_corr(cond):
        """Split-half correlation (rho) from a radial condition dict, else NaN."""
        if isinstance(cond, dict) and cond.get('reliable_corr') is not None:
            return float(cond['reliable_corr'])
        return np.nan

    def _radial_reliability(self, df):
        """Normalize the radial df to columns: cell, distance_reliable, distance_corr, + context."""
        cell_col = 'cell_id' if 'cell_id' in df.columns else 'cell'
        out = pd.DataFrame({'cell': df[cell_col].astype(int).values})

        cond_cols = [c for c in ('monocular', 'binocular') if c in df.columns]
        if cond_cols:
            out['distance_reliable'] = df.apply(
                lambda r: any(self._cond_reliable(r[c]) for c in cond_cols), axis=1
            ).values

            def _best_corr(row):
                vals = [self._cond_corr(row[c]) for c in cond_cols]
                vals = [v for v in vals if not np.isnan(v)]
                return max(vals) if vals else np.nan
            out['distance_corr'] = df.apply(_best_corr, axis=1).values   # best rho across conditions
        elif 'reliable' in df.columns:
            out['distance_reliable'] = df['reliable'].astype(bool).values
            if 'mean_corr' in df.columns:
                out['distance_corr'] = df['mean_corr'].values
        else:
            raise KeyError("radial df has no 'monocular'/'binocular' condition dicts nor a "
                           "'reliable' column to derive distance reliability from.")

        for src_col, new_col in (('RF', 'distance_RF'), ('distance', 'pref_distance'), ('ODI', 'distance_ODI')):
            if src_col in df.columns:
                out[new_col] = df[src_col].values
        return out

    def _disparity_reliability(self, df):
        """Normalize the disparity df to columns: cell, goggles_reliable, + context."""
        out = pd.DataFrame({'cell': df['cell'].astype(int).values})

        eye_cols = [c for c in ('right_reliable', 'left_reliable') if c in df.columns]
        if eye_cols:
            out['goggles_reliable'] = df[eye_cols].any(axis=1).values   # either eye
        elif 'reliable' in df.columns:
            out['goggles_reliable'] = df['reliable'].astype(bool).values
        else:
            raise KeyError("disparity df has no 'right_reliable'/'left_reliable' nor 'reliable' column.")

        # carry the per-eye reliable flags (for per-eye selection)
        for col in eye_cols:
            out[col] = df[col].astype(bool).values

        # carry the per-eye split-half correlations (rho) for thresholding
        for col in ('right_mean_corr', 'left_mean_corr', 'mean_corr'):
            if col in df.columns:
                out[col] = df[col].values

        for col in ('right_px', 'left_px', 'px'):
            if col in df.columns:
                out[col] = df[col].values
        if 'ODI' in df.columns:
            out['goggles_ODI'] = df['ODI'].values
        return out

    # ---------------------------------------------------------------- selection

    def merged_table(self):
        """Per common-cell table with the reliability flags + carried context."""
        return self.merged

    def _ids(self, mask):
        return np.sort(self.merged.loc[mask, 'cell'].to_numpy())

    def reliable_distance(self):
        return self._ids(self.merged['distance_reliable'])

    def reliable_goggles(self):
        return self._ids(self.merged['goggles_reliable'])

    def reliable_both(self):
        return self._ids(self.merged['reliable_both'])

    def reliable_either(self):
        return self._ids(self.merged['distance_reliable'] | self.merged['goggles_reliable'])

    # ------------------------------------------------------------------ summary

    @staticmethod
    def _pct(k, n):
        return f"{100 * k / n:5.1f}%" if n else "  n/a"

    def print_summary(self):
        """Print reliable-cell counts across the two experiments."""
        m = self.merged
        n = len(m)
        d = int(m['distance_reliable'].sum())
        g = int(m['goggles_reliable'].sum())
        both = int(m['reliable_both'].sum())
        either = int((m['distance_reliable'] | m['goggles_reliable']).sum())
        line = "=" * 52

        print(line)
        print(f" Common cells: {n}")
        print("-" * 52)
        print(f" {'Distance reliable':24s}: {d:4d}  ({self._pct(d, n)})")
        print(f" {'Goggles reliable':24s}: {g:4d}  ({self._pct(g, n)})")
        print(f" {'Reliable in BOTH':24s}: {both:4d}  ({self._pct(both, n)})")
        print("-" * 52)
        print(f" {'Distance-only':24s}: {d - both:4d}")
        print(f" {'Goggles-only':24s}: {g - both:4d}")
        print(f" {'Either':24s}: {either:4d}  ({self._pct(either, n)})")
        print(f" {'Neither':24s}: {n - either:4d}  ({self._pct(n - either, n)})")
        print(line)

        return {'common': n, 'distance': d, 'goggles': g, 'both': both, 'either': either}

    def plot_overlap(self):
        """Bar chart of distance-only / goggles-only / both / neither counts."""
        m = self.merged
        n = len(m)
        both = int(m['reliable_both'].sum())
        d_only = int(m['distance_reliable'].sum()) - both
        g_only = int(m['goggles_reliable'].sum()) - both
        neither = n - (both + d_only + g_only)

        labels = ['Distance\nonly', 'Goggles\nonly', 'Both', 'Neither']
        counts = [d_only, g_only, both, neither]
        colors = ['#31688e', 'red', 'purple', 'lightgray']

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c),
                    ha='center', va='bottom', fontsize=12)
        ax.set_ylabel('Number of cells')
        ax.set_title(f'Reliability overlap (n={n} common cells)')
        plt.tight_layout()
        return fig

    def plot_rf_scatter(self, eye='right', cells=None, corr_threshold=None, fit=True,
                        xlim=None, ylim=None):
        """
        Scatter of distance-experiment RF angle (x, deg) vs goggles RF position
        (y, px).

        By default uses cells reliable in BOTH the distance experiment and
        ``eye``. Pass ``cells`` to instead scatter a hand-chosen group.

        Parameters
        ----------
        eye : 'right' or 'left'
            Which eye's px (and, when ``cells`` is None, reliability) to use.
        cells : sequence of int, optional
            Explicit cell ids to plot. Overrides the reliability selection; only
            ids present in the merged table (common to both sessions) are kept.
            ``corr_threshold`` still applies on top if also given.
        corr_threshold : float, optional
            If given, additionally require the split-half correlation (rho) to be
            >= this value in BOTH experiments (the eye's ``{eye}_mean_corr`` and
            the radial ``distance_corr``).
        fit : bool
            Overlay a linear fit (slope + Pearson r).
        """
        m = self.merged
        rel_col = f'{eye}_reliable' if f'{eye}_reliable' in m.columns else 'goggles_reliable'
        px_col = f'{eye}_px' if f'{eye}_px' in m.columns else 'px'
        if 'distance_RF' not in m.columns or px_col not in m.columns:
            raise KeyError("merged table needs 'distance_RF' (radial) and the eye's px column "
                           f"('{px_col}'); regenerate the DataFrames if missing.")

        if cells is not None:
            cells = [int(c) for c in cells]
            n_missing = len(set(cells) - set(m['cell'].astype(int)))
            if n_missing:
                print(f"[note] {n_missing} of {len(cells)} requested cells are not in the merged "
                      "table (not common to both sessions); skipped.")
            mask = m['cell'].astype(int).isin(cells)
        else:
            # require reliability in BOTH the distance experiment and this eye
            mask = m[rel_col].astype(bool) & m['distance_reliable'].astype(bool)

        # optional rho threshold on top of the boolean reliable flags
        if corr_threshold is not None:
            eye_corr_col = f'{eye}_mean_corr' if f'{eye}_mean_corr' in m.columns else 'mean_corr'
            have_eye = eye_corr_col in m.columns
            have_dist = 'distance_corr' in m.columns
            if not have_eye and not have_dist:
                raise KeyError(
                    "corr_threshold was set, but the merged table has no correlation columns "
                    f"('distance_corr', '{eye_corr_col}'). This usually means `ana` is stale — "
                    "rebuild it: CrossExperimentAnalysis.from_devices(R, D).")
            if have_eye:
                mask &= m[eye_corr_col] >= corr_threshold
            else:
                print(f"[warning] no '{eye_corr_col}' column; rho threshold applied to distance only.")
            if have_dist:
                mask &= m['distance_corr'] >= corr_threshold
            else:
                print("[warning] no 'distance_corr' column; rho threshold applied to goggles only.")

        sub = m.loc[mask, ['cell', 'distance_RF', px_col]].dropna()

        color = {'right': 'blue', 'left': 'red'}.get(eye, 'black')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub['distance_RF'], sub[px_col], color=color, edgecolor='black', alpha=0.8)
        if cells is not None:   # only label points for a hand-picked group
            for _, row in sub.iterrows():
                ax.annotate(str(int(row['cell'])), (row['distance_RF'], row[px_col]),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 2), textcoords='offset points')

        if fit and len(sub) >= 2:
            x = sub['distance_RF'].to_numpy(float)
            y = sub[px_col].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, color='black', linewidth=2,
                    label=f'slope={slope:.2f}, r={r:.2f}')
            ax.legend(frameon=False, fontsize=12)

        rho_txt = f', ρ≥{corr_threshold}' if corr_threshold is not None else ''
        base = 'Hand-picked cells' if cells is not None else f'Reliable in distance & {eye} eye'
        ax.set_xlabel('Distance RF (deg)')
        ax.set_ylabel(f'{eye.capitalize()}-eye RF (px)')
        ax.set_title(f'{base}{rho_txt} (n={len(sub)})')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()
        return fig

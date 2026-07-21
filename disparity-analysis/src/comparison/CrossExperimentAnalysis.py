"""
Cross-experiment reliability across up to three experiments run on the same
suite2p split: the radial (distance) experiment, the per-eye monocular goggles
experiment, and the binocular disparity-stimulus experiment.

Works purely on the per-cell stats DataFrames that each experiment's analyzer
saves (and ``save_dataframe()`` persists). Reading DataFrames needs only pandas,
so this never imports either experiment's ``src`` package — no namespace
collision between ``radial-analysis/src`` and ``disparity-analysis/src``.

Naming (radial / monocular / disparity):
  - radial (distance): reliable in any available monocular/binocular condition
    (radial's split-half-vs-null + control-distance check).
  - monocular (per-eye goggles, ``MonocularDevice``): reliable in either eye
    (or the single eye present).
  - disparity (``DisparityDevice``): reliable in ANY disparity — collapsed
    from that experiment's long (one row per cell x disparity) df.

The disparity experiment is optional; with only the first two this behaves
exactly like the original two-way analysis. Cell ids correspond only for a
same-suite2p-run split; ``__init__`` warns if the overlap is empty.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CrossExperimentAnalysis:

    # per-eye colours, matching the viz layer's EYE_COLOR / EYE_CMAP
    EYE_COLOR = {'left': 'red', 'right': 'blue'}
    EYE_CMAP = {'left': 'Reds', 'right': 'Blues'}

    def __init__(self, radial_df, monocular_df, disparity_df=None, how='inner'):
        self.radial_df = radial_df
        self.monocular_df = monocular_df
        self.disparity_df = disparity_df

        rad = self._radial_reliability(radial_df)
        mono = self._monocular_reliability(monocular_df)

        self.merged = pd.merge(rad, mono, on='cell', how=how)
        self.merged['reliable_both'] = (
            self.merged['distance_reliable'] & self.merged['monocular_reliable']
        )

        if disparity_df is not None:
            disp = self._disparity_reliability(disparity_df)
            self.merged = pd.merge(self.merged, disp, on='cell', how=how)
            self.merged['reliable_all'] = (
                self.merged['distance_reliable']
                & self.merged['monocular_reliable']
                & self.merged['disparity_reliable']
            )

        if len(self.merged) == 0:
            print("[warning] no common cells between the DataFrames — the sessions "
                  "may not be a same-suite2p-run split pair, so cell ids don't correspond.")

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_paths(cls, radial_path, monocular_path, disparity_path=None, **kwargs):
        """Load the per-cell stats DataFrames from pickle files (disparity optional)."""
        disparity_df = pd.read_pickle(disparity_path) if disparity_path is not None else None
        return cls(pd.read_pickle(radial_path), pd.read_pickle(monocular_path),
                   disparity_df, **kwargs)

    @classmethod
    def from_devices(cls, distance_device, monocular_device, disparity_device=None, **kwargs):
        """Load each device's default ``{session_name}_dataframe.pkl`` from its data_dir."""
        def _path(device):
            return os.path.join(device.data_dir, f"{device.session_name}_dataframe.pkl")
        disparity_path = _path(disparity_device) if disparity_device is not None else None
        return cls.from_paths(_path(distance_device), _path(monocular_device),
                              disparity_path, **kwargs)

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

    def _monocular_reliability(self, df):
        """Normalize the monocular (per-eye goggles) df to: cell, monocular_reliable, + context."""
        out = pd.DataFrame({'cell': df['cell'].astype(int).values})

        eye_cols = [c for c in ('right_reliable', 'left_reliable') if c in df.columns]
        if eye_cols:
            out['monocular_reliable'] = df[eye_cols].any(axis=1).values   # either eye
        elif 'reliable' in df.columns:
            out['monocular_reliable'] = df['reliable'].astype(bool).values
        else:
            raise KeyError("monocular df has no 'right_reliable'/'left_reliable' nor 'reliable' column.")

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
            out['monocular_ODI'] = df['ODI'].values
        return out

    def _disparity_reliability(self, df):
        """
        Collapse the disparity-stim df to one row per cell.

        That experiment's df is long (one row per cell x disparity). A cell counts
        as ``disparity_reliable`` if it is reliable at ANY disparity; ``disparity_corr``
        is its best (max) split-half rho and ``disparity_best_disp`` the disparity
        where that best rho occurs.
        """
        if 'disparity' in df.columns:                     # long form
            rel = df.groupby('cell')['reliable'].any()
            out = pd.DataFrame({
                'cell': rel.index.astype(int),
                'disparity_reliable': rel.values,
            })
            if 'mean_corr' in df.columns:
                out['disparity_corr'] = df.groupby('cell')['mean_corr'].max().values
                best = df.loc[df.groupby('cell')['mean_corr'].idxmax()]
                out['disparity_best_disp'] = best['disparity'].astype(int).values
            return out

        # flat fallback: already one row per cell
        out = pd.DataFrame({'cell': df['cell'].astype(int).values})
        if 'reliable' in df.columns:
            out['disparity_reliable'] = df['reliable'].astype(bool).values
        else:
            raise KeyError("disparity df has no 'disparity'/'reliable' columns to collapse.")
        if 'mean_corr' in df.columns:
            out['disparity_corr'] = df['mean_corr'].values
        return out

    # ---------------------------------------------------------------- selection

    @property
    def has_disparity(self):
        """True when the disparity (DisparityDevice) experiment was provided."""
        return 'disparity_reliable' in self.merged.columns

    def merged_table(self):
        """Per common-cell table with the reliability flags + carried context."""
        return self.merged

    def _ids(self, mask):
        return np.sort(self.merged.loc[mask, 'cell'].to_numpy())

    def reliable_distance(self):
        return self._ids(self.merged['distance_reliable'])

    def reliable_monocular(self):
        return self._ids(self.merged['monocular_reliable'])

    def reliable_both(self):
        """Cells reliable in the distance AND monocular experiments."""
        return self._ids(self.merged['reliable_both'])

    def reliable_either(self):
        return self._ids(self.merged['distance_reliable'] | self.merged['monocular_reliable'])

    def reliable_disparity(self):
        if not self.has_disparity:
            raise AttributeError("no disparity experiment was provided (pass disparity_df).")
        return self._ids(self.merged['disparity_reliable'])

    def reliable_all(self):
        """Cells reliable in all three experiments (requires the disparity df)."""
        if not self.has_disparity:
            raise AttributeError("reliable_all() needs the disparity experiment; pass disparity_df.")
        return self._ids(self.merged['reliable_all'])

    def reliable_any(self):
        """Cells reliable in at least one experiment."""
        mask = self.merged['distance_reliable'] | self.merged['monocular_reliable']
        if self.has_disparity:
            mask = mask | self.merged['disparity_reliable']
        return self._ids(mask)

    # ------------------------------------------------------------------ summary

    @staticmethod
    def _pct(k, n):
        return f"{100 * k / n:5.1f}%" if n else "  n/a"

    @staticmethod
    def _corr_thresh(corr_threshold, key):
        """
        Resolve a per-condition rho threshold. ``corr_threshold`` is either a
        scalar (applied to every condition) or a dict keyed by 'distance' /
        'left' / 'right' (a missing key means no threshold for that condition).
        """
        if corr_threshold is None:
            return None
        if isinstance(corr_threshold, dict):
            return corr_threshold.get(key)
        return corr_threshold

    @staticmethod
    def _rho_txt(corr_threshold):
        """Title annotation describing the applied rho threshold(s)."""
        if corr_threshold is None:
            return ''
        if isinstance(corr_threshold, dict):
            inner = ', '.join(f'{k}:{v}' for k, v in corr_threshold.items())
            return f', ρ≥{{{inner}}}'
        return f', ρ≥{corr_threshold}'

    def print_summary(self):
        """Print reliable-cell counts across the experiments (2- or 3-way)."""
        m = self.merged
        n = len(m)
        dist = int(m['distance_reliable'].sum())
        mono = int(m['monocular_reliable'].sum())
        both = int(m['reliable_both'].sum())
        line = "=" * 52

        print(line)
        print(f" Common cells: {n}")
        print("-" * 52)
        print(f" {'Distance reliable':26s}: {dist:4d}  ({self._pct(dist, n)})")
        print(f" {'Monocular reliable':26s}: {mono:4d}  ({self._pct(mono, n)})")

        out = {'common': n, 'distance': dist, 'monocular': mono, 'both': both}

        if self.has_disparity:
            disp = int(m['disparity_reliable'].sum())
            allthree = int(m['reliable_all'].sum())
            any_ = int((m['distance_reliable'] | m['monocular_reliable'] | m['disparity_reliable']).sum())
            print(f" {'Disparity reliable':26s}: {disp:4d}  ({self._pct(disp, n)})")
            print("-" * 52)
            print(f" {'Distance & monocular':26s}: {both:4d}  ({self._pct(both, n)})")
            print(f" {'Reliable in ALL three':26s}: {allthree:4d}  ({self._pct(allthree, n)})")
            print(f" {'Reliable in ANY':26s}: {any_:4d}  ({self._pct(any_, n)})")
            print(f" {'None':26s}: {n - any_:4d}  ({self._pct(n - any_, n)})")
            print(line)
            out.update({'disparity': disp, 'all': allthree, 'any': any_})
            return out

        either = int((m['distance_reliable'] | m['monocular_reliable']).sum())
        print(f" {'Reliable in BOTH':26s}: {both:4d}  ({self._pct(both, n)})")
        print("-" * 52)
        print(f" {'Distance-only':26s}: {dist - both:4d}")
        print(f" {'Monocular-only':26s}: {mono - both:4d}")
        print(f" {'Either':26s}: {either:4d}  ({self._pct(either, n)})")
        print(f" {'Neither':26s}: {n - either:4d}  ({self._pct(n - either, n)})")
        print(line)
        out['either'] = either
        return out

    def plot_overlap(self):
        """Bar chart of the disjoint reliability regions (2- or 3-way)."""
        m = self.merged
        n = len(m)
        dist = m['distance_reliable'].astype(bool)
        mono = m['monocular_reliable'].astype(bool)

        if self.has_disparity:
            disp = m['disparity_reliable'].astype(bool)
            regions = {
                'Distance\nonly':  int((dist & ~mono & ~disp).sum()),
                'Monocular\nonly': int((~dist & mono & ~disp).sum()),
                'Disparity\nonly': int((~dist & ~mono & disp).sum()),
                'Dist &\nMono':    int((dist & mono & ~disp).sum()),
                'Dist &\nDisp':    int((dist & ~mono & disp).sum()),
                'Mono &\nDisp':    int((~dist & mono & disp).sum()),
                'All\nthree':      int((dist & mono & disp).sum()),
                'None':            int((~dist & ~mono & ~disp).sum()),
            }
            colors = ['#31688e', 'red', '#35b779', '#8856a7', '#2c7fb8',
                      '#d95f0e', 'purple', 'lightgray']
        else:
            regions = {
                'Distance\nonly':  int((dist & ~mono).sum()),
                'Monocular\nonly': int((~dist & mono).sum()),
                'Both':            int((dist & mono).sum()),
                'Neither':         int((~dist & ~mono).sum()),
            }
            colors = ['#31688e', 'red', 'purple', 'lightgray']

        labels = list(regions.keys())
        counts = list(regions.values())
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 4))
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c),
                    ha='center', va='bottom', fontsize=12)
        ax.set_ylabel('Number of cells')
        ax.set_title(f'Reliability overlap (n={n} common cells)')
        plt.tight_layout()
        return fig

    def reliable_each_thresh(self, corr_threshold, monocular='any'):
        """
        Reliable cells in EACH experiment separately, at split-half rho >= corr_threshold.

        Complements ``plot_overlap()``: that shows the disjoint overlap regions,
        this shows each experiment's own reliable count with its own rho applied
        (NOT the all-three intersection — a cell can appear in more than one set).

        Parameters
        ----------
        corr_threshold : float or dict
            Split-half rho floor, applied independently within each experiment
            (the distance session's own corr column, the eye's own mean_corr,
            the disparity session's best-disparity corr). A scalar applies the
            same floor everywhere; a dict keyed by 'distance' / 'left' / 'right'
            / 'monocular' (single-eye sessions) / 'disparity' sets them
            independently — a missing key means no threshold for that condition,
            same convention as ``_corr_thresh``.
        monocular : {'any', 'left', 'right', 'split', 'all'}
            How to express the monocular (per-eye goggles) experiment:
              'any'   -> one 'monocular' set: reliable in the left OR the right eye
              'left'  -> just 'left'
              'right' -> just 'right'
              'split' -> both 'left' and 'right' as separate sets
              'all'   -> 'left', 'right', and 'monocular' (any)

        Returns
        -------
        dict of sorted ndarray, always with 'distance' and 'disparity' (when this
        experiment has one), plus the monocular key(s) selected above.
        """
        m = self.merged
        ids = lambda mask: np.sort(m.loc[mask, 'cell'].to_numpy())

        def _ge(col, key):
            t = self._corr_thresh(corr_threshold, key)
            return True if t is None else (m[col] >= t)

        out = {'distance': ids(m['distance_reliable'].astype(bool) & _ge('distance_corr', 'distance'))}

        both_eyes = {'right_reliable', 'left_reliable'}.issubset(m.columns)
        if both_eyes:
            left = m['left_reliable'].astype(bool) & _ge('left_mean_corr', 'left')
            right = m['right_reliable'].astype(bool) & _ge('right_mean_corr', 'right')
        else:                                                # single-eye session: no L/R split
            any_ = m['monocular_reliable'].astype(bool) & _ge('mean_corr', 'monocular')
            left = right = any_

        want = {'any': ['monocular'], 'left': ['left'], 'right': ['right'],
                'split': ['left', 'right'], 'all': ['left', 'right', 'monocular']}[monocular]
        masks = {'left': left, 'right': right, 'monocular': left | right}
        for key in want:
            out[key] = ids(masks[key])

        if self.has_disparity:
            out['disparity'] = ids(m['disparity_reliable'].astype(bool) & _ge('disparity_corr', 'disparity'))
        return out

    def plot_reliable_each(self, corr_threshold, monocular='all', ax=None):
        """Bar chart of ``reliable_each_thresh``'s per-experiment counts."""
        sets = self.reliable_each_thresh(corr_threshold, monocular=monocular)
        colors = {'distance': '#31688e', 'left': 'red', 'right': 'blue',
                  'monocular': '#8856a7', 'disparity': '#35b779'}
        labels = {'monocular': 'monocular\n(any eye)'}

        if ax is None:
            _, ax = plt.subplots(figsize=(max(5, 1.3 * len(sets)), 4))
        keys = list(sets)
        counts = [len(sets[k]) for k in keys]
        bars = ax.bar(keys, counts, color=[colors.get(k, 'gray') for k in keys],
                      edgecolor='black', alpha=0.85)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c),
                    ha='center', va='bottom', fontsize=12)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels.get(k, k) for k in keys])
        ax.set_ylabel('Reliable cells')
        ax.set_title(f'Reliable per experiment (ρ≥{corr_threshold}, n={len(self.merged)} cells)')
        return ax.figure

    # ------------------------------------------------------------- calibration

    def _calibration_subset(self, eye, cells=None, corr_threshold=None, signed=True):
        """
        Cells carrying both a distance RF angle (deg) and this eye's goggles RF pixel.

        Returns a DataFrame with columns ``cell``, ``distance_RF``, ``px`` — the
        (theta, pixel) pairs the screen calibration regresses — plus ``corr``,
        that eye's split-half rho, when the merged table carries it. ``corr`` is
        carried for weighting/display only and never gates a row: rows are dropped
        on missing ``distance_RF``/``px`` alone, so the fitted n does not depend on
        whether a rho happens to be recorded.

        ``signed`` puts both eyes on the stimulus' signed joint axis (left panel
        negative, right panel positive, meeting at a nasal edge of 0) — the
        convention ``teensyStimulusAngle`` expects. The monocular df stores the
        left eye's *drawn panel* pixel, which is the negation of its signed-axis
        pixel, so the left eye is negated here and the right passed through.
        With ``signed=False`` the raw dataframe pixels come back untouched.
        """
        m = self.merged
        rel_col = f'{eye}_reliable' if f'{eye}_reliable' in m.columns else 'monocular_reliable'
        px_col = f'{eye}_px' if f'{eye}_px' in m.columns else 'px'
        corr_col = f'{eye}_mean_corr' if f'{eye}_mean_corr' in m.columns else 'mean_corr'
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
            t_dist = self._corr_thresh(corr_threshold, 'distance')
            t_eye = self._corr_thresh(corr_threshold, eye)
            have_eye = eye_corr_col in m.columns
            have_dist = 'distance_corr' in m.columns
            if not have_eye and not have_dist:
                raise KeyError(
                    "corr_threshold was set, but the merged table has no correlation columns "
                    f"('distance_corr', '{eye_corr_col}'). This usually means `ana` is stale — "
                    "rebuild it: CrossExperimentAnalysis.from_devices(R, D).")
            if t_eye is not None:
                if have_eye:
                    mask &= m[eye_corr_col] >= t_eye
                else:
                    print(f"[warning] no '{eye_corr_col}' column; rho threshold applied to distance only.")
            if t_dist is not None:
                if have_dist:
                    mask &= m['distance_corr'] >= t_dist
                else:
                    print("[warning] no 'distance_corr' column; rho threshold applied to goggles only.")

        cols = ['cell', 'distance_RF', px_col]
        if corr_col in m.columns:
            cols.append(corr_col)
        sub = m.loc[mask, cols].dropna(subset=['distance_RF', px_col])
        sub = sub.rename(columns={px_col: 'px', corr_col: 'corr'})
        if signed and eye == 'left':
            sub['px'] = -sub['px']
        return sub

    def fit_eye_calibration(self, eye='right', cells=None, corr_threshold=None,
                            signed=True, method='ols'):
        """
        Recover one eye's screen calibration: regress goggles RF pixel on the
        distance experiment's RF angle across the cells the two sessions share.

        Each cell is a fixed azimuth landmark — the distance experiment measures
        its true angle with a physical stimulus, independently of any screen — so
        the recovered slope/intercept are the deg->pixel mapping that eye's screen
        actually realises. With ``signed=True`` they are in ``teensyStimulusAngle``
        argument convention.

        Parameters
        ----------
        method : 'ols' or 'odr'
            ``'ols'`` regresses px on angle (``np.polyfit``), which assumes the
            angle is known exactly. It is not: it is measured and quantized to the
            sweep's per-frame step, so OLS attenuates the slope toward zero.
            ``'odr'`` runs orthogonal distance regression (errors in both
            variables), seeded from the OLS fit. Compare the two — the gap is the
            attenuation.

        Returns
        -------
        dict with ``slope``, ``intercept``, ``r`` (Pearson), ``n``, ``deg_min``,
        ``deg_max``, ``eye``, ``method``, ``signed``. ``slope``/``intercept`` are
        NaN when fewer than 2 cells survive selection.

        Note
        ----
        ``r`` is the number to read first. A slope means nothing if the pixel and
        the angle are not actually related; an eye that drives this hemisphere
        weakly can yield a confident-looking slope over a handful of cells at
        r ~ 0.
        """
        sub = self._calibration_subset(eye, cells=cells, corr_threshold=corr_threshold,
                                       signed=signed)
        x = sub['distance_RF'].to_numpy(float)
        y = sub['px'].to_numpy(float)
        out = {'eye': eye, 'method': method, 'signed': signed, 'n': len(sub),
               'slope': np.nan, 'intercept': np.nan, 'r': np.nan,
               'deg_min': np.nan, 'deg_max': np.nan}
        if len(sub) < 2:
            return out

        slope, intercept = np.polyfit(x, y, 1)       # OLS, and the ODR seed
        if method == 'odr':
            from scipy import odr                    # lazy: keeps the module pandas-only
            fit = odr.ODR(odr.RealData(x, y),
                          odr.Model(lambda B, xx: B[0] * xx + B[1]),
                          beta0=[slope, intercept]).run()
            slope, intercept = float(fit.beta[0]), float(fit.beta[1])
        elif method != 'ols':
            raise ValueError("method must be 'ols' or 'odr'")

        out.update({'slope': float(slope), 'intercept': float(intercept),
                    'r': float(np.corrcoef(x, y)[0, 1]),
                    'deg_min': float(x.min()), 'deg_max': float(x.max())})
        return out

    def _draw_stim_reference(self, ax, eye, stim_device, color, sub=None, band_color=None):
        """
        Draw what the session *assumed*, as a reference for what was recovered.

        Two things, both read off ``stim_device``'s stimulus log:

        - the **assumed** deg->pixel line the session was actually run with. It needs
          no transform: the stim fits are defined on the same signed joint axis these
          plots use, so ``slope*deg + intercept`` lands directly on the y axis.
        - the **coverage band**: the pixels that actually reached that eye's panel.
          The sweep can run past the nasal edge (0), and anything beyond it is off-panel
          and drawn gray, so the band is the swept range clipped to the eye's own half
          of the axis. No cell can have an RF outside its band — the pixel was never shown.

        ``stim_device`` is duck-typed (``{eye}_fit``, ``parameters``), never imported,
        so this module stays pandas-only — same contract as ``from_devices``.

        ``sub`` is that eye's calibration subset; when given, the assumed line is
        scored against it with ``_line_r2`` so it can be read against the recovered
        line's R2. Read the two R2 values together: they say how much of the pixel
        spread each line accounts for, which is the question "was the assumption
        good enough to keep?".

        ``color`` draws the line, ``band_color`` the coverage band (defaulting to
        ``color``). They separate so a caller shading dots with a same-hue colormap
        can contrast the line without recolouring the band, which reads better in
        the eye's own colour.

        Returns the ``(lo, hi)`` band drawn, or None when the session's log carries no
        fit for this eye (the legacy ``goggles_left`` format predates them).
        """
        fit = getattr(stim_device, f'{eye}_fit', None)
        if fit is None:
            return None

        p = stim_device.parameters
        slope, intercept = float(fit[0]), float(fit[1])
        xs = np.linspace(float(p['start_deg']), float(p['end_deg']), 100)
        px = slope * xs + intercept

        label = f'{eye.capitalize()} assumed: px = {slope:.2f}·deg {intercept:+.1f}'
        if sub is not None and len(sub) >= 2:
            r2 = self._line_r2(sub['distance_RF'], sub['px'], slope, intercept)
            label += f'  (R²={r2:.2f})'
        ax.plot(xs, px, color=color, linestyle='--', linewidth=2, alpha=0.75, zorder=2,
                label=label)

        # clip the sweep to this eye's half of the joint axis: left [-W, 0], right [0, +W]
        width = float(p['W'])
        lo, hi = (max(-width, px.min()), min(0.0, px.max())) if eye == 'left' \
            else (max(0.0, px.min()), min(width, px.max()))
        if hi <= lo:
            return None                     # the sweep never landed on this panel
        ax.axhspan(lo, hi, color=band_color or color, alpha=0.08, zorder=0)
        return lo, hi

    @staticmethod
    def _expand_ylim(ylim, bands):
        """Grow ylim to contain every band drawn — never draw a band the axis hides."""
        vals = [v for band in bands if band for v in band]
        if not vals:
            return ylim
        return min(ylim[0], min(vals) - 5), max(ylim[1], max(vals) + 5)

    @staticmethod
    def _line_r2(x, y, slope, intercept):
        """
        Variance of y explained by a **given** line — not by a fitted one.

        Pearson r cannot judge a fixed line: ``corr(m*x + b, y) == ±corr(x, y)``
        for any ``m != 0``, so it returns the same value however wrong ``m`` and
        ``b`` are — blind to precisely what an assumed fit needs testing for.
        R2 measured against that line's own residuals does judge it, and goes
        **negative** when the line predicts worse than the mean of y.

        For an OLS fit this equals r**2; for the assumed line it generally does not.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    def plot_rf_scatter(self, eye='right', cells=None, corr_threshold=None, fit=True,
                        xlim=None, ylim=None, signed=True, stim_device=None,
                        color_by_corr=False, cmap=None, clim=None):
        """
        Scatter of distance-experiment RF angle (x, deg) vs goggles RF position
        (y, px), with the calibration fit overlaid.

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
        corr_threshold : float or dict, optional
            Require the split-half correlation (rho) to be >= this value. A scalar
            applies to both the eye and the distance experiment; a dict keyed by
            'distance' / 'left' / 'right' sets them independently.
        fit : bool
            Overlay the linear fit (slope + Pearson r).
        signed : bool
            Plot on the signed joint axis (left negative, right positive), which
            is what the default ``ylim`` assumes. See ``_calibration_subset``.
        stim_device : goggles device, optional
            The session whose pixels are on the y axis (i.e. the **monocular**
            device). When given, overlays the deg->pixel fit that session was
            actually run with, plus the band of pixels that reached the screen —
            so the recovered fit can be read against the assumed one. See
            ``_draw_stim_reference``.
        color_by_corr : bool
            Shade each dot by that cell's split-half rho in this eye, with a
            colourbar. Lets the fit be judged by eye: a line is only worth as much
            as the dark dots it passes through, and a slope carried by pale ones is
            being set by cells that barely responded.
        cmap : str, optional
            Colormap for ``color_by_corr``. Defaults per eye via ``EYE_CMAP`` —
            'Reds' for left, 'Blues' for right — matching the red/blue eye colours
            used everywhere else, so the eye stays readable from the plot alone.
        clim : (lo, hi), optional
            Colour limits for ``color_by_corr``. Defaults to the rho range actually
            admitted: ``(corr_threshold_for_this_eye, 1.0)``, or ``(min rho present,
            1.0)`` when no threshold is set — a 0..1 ramp would spend half its range
            on shades no cell can have, since selection already requires
            ``{eye}_reliable``. Both eyes share these limits at a given threshold,
            so left and right stay comparable; plots at *different* thresholds do
            not, as the same shade then means a different rho. Pass an explicit
            ``(lo, hi)`` to pin it (e.g. ``(0, 1)`` for an absolute scale).
        """
        sub = self._calibration_subset(eye, cells=cells, corr_threshold=corr_threshold,
                                       signed=signed)

        color = self.EYE_COLOR.get(eye, 'black')
        cmap = cmap or self.EYE_CMAP.get(eye, 'viridis')
        fig, ax = plt.subplots(figsize=(6, 6))

        shade = color_by_corr and 'corr' in sub.columns
        if color_by_corr and not shade:
            print(f"[warning] merged table has no '{eye}_mean_corr'; "
                  "drawing uniform dot colour.")
        if shade:
            if clim is None:
                t_eye = self._corr_thresh(corr_threshold, eye)
                if t_eye is not None:
                    lo = float(t_eye)          # the bar these cells had to clear
                elif len(sub):
                    lo = float(sub['corr'].min())
                else:
                    lo = 0.5
                clim = (min(lo, 0.99), 1.0)    # guard a degenerate vmin == vmax
            sc = ax.scatter(sub['distance_RF'], sub['px'], c=sub['corr'], cmap=cmap,
                            vmin=clim[0], vmax=clim[1], edgecolor='black',
                            alpha=0.9, zorder=3)
            fig.colorbar(sc, ax=ax, label='split-half ρ', fraction=0.046, pad=0.04)
        else:
            ax.scatter(sub['distance_RF'], sub['px'], color=color,
                       edgecolor='black', alpha=0.8)

        band = None
        if stim_device is not None and signed:
            # the eye colour would vanish into a same-hue colormap, so contrast the
            # reference LINE when the dots are shaded — the band stays the eye's
            # colour either way, since at alpha 0.08 behind the dots it never clashes
            line_color = 'darkorange' if shade else color
            band = self._draw_stim_reference(ax, eye, stim_device, line_color,
                                             sub=sub, band_color=color)
        if cells is not None:   # only label points for a hand-picked group
            for _, row in sub.iterrows():
                ax.annotate(str(int(row['cell'])), (row['distance_RF'], row['px']),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 2), textcoords='offset points')

        if fit and len(sub) >= 2:
            f = self.fit_eye_calibration(eye, cells=cells, corr_threshold=corr_threshold,
                                         signed=signed)
            xs = np.linspace(f['deg_min'], f['deg_max'], 100)
            # same 'px = m·deg + b' form as the assumed line, and scored the same
            # way, so the two are read against each other rather than by eye
            r2 = self._line_r2(sub['distance_RF'], sub['px'], f['slope'], f['intercept'])
            ax.plot(xs, f['slope'] * xs + f['intercept'], color='black', linewidth=2,
                    zorder=4,
                    label=f"{eye.capitalize()} fit: px = {f['slope']:.2f}·deg "
                          f"{f['intercept']:+.1f}  (r={f['r']:.2f}, R²={r2:.2f})")
            ax.legend(frameon=False, fontsize=10, loc='best')

        rho_txt = self._rho_txt(corr_threshold)
        base = 'Hand-picked cells' if cells is not None else f'Reliable in distance & {eye} eye'
        ax.set_xlabel('Distance RF (deg)')
        ax.set_ylabel(f'{eye.capitalize()}-eye RF (px)')
        ax.set_title(f'{base}{rho_txt} (n={len(sub)})')
        if xlim is not None:
            ax.set_xlim(xlim)
        # default to the eye's full screen range on the signed axis
        # (right: 0..120, left: -120..0); raw px is positive for both eyes
        if ylim is None:
            ylim = (0, 120) if (eye == 'right' or not signed) else (-120, 0)
        ax.set_ylim(self._expand_ylim(ylim, [band]))
        plt.tight_layout()
        return fig

    def plot_rf_scatter_both(self, cells=None, corr_threshold=None, fit=True,
                             xlim=None, ylim=(-120, 120), stim_device=None):
        """
        Both eyes on one axis, for cells reliable in distance and in >=1 eye.

        Each cell sits at one x (its distance-experiment RF angle, deg). A cell
        reliable in BOTH eyes contributes two points -- left-eye RF (red,
        negative px) and right-eye RF (blue, positive px) -- joined by a vertical
        line whose span is that cell's left/right RF disparity. A cell reliable
        in only ONE eye contributes a single point for that eye, with no
        connector.

        Both eyes are drawn on the signed joint axis (see ``_calibration_subset``),
        so the left/right span of a connector is a real disparity in stimulus
        units rather than an artefact of the two panels' pixel conventions.

        Parameters
        ----------
        cells : sequence of int, optional
            Explicit cell ids to plot. Overrides the reliability selection (both
            eyes' points are drawn and connected); only ids present in the merged
            table are kept. ``corr_threshold`` still applies on top if also given.
        corr_threshold : float or dict, optional
            Require the split-half correlation (rho) to be >= this value: in the
            distance experiment for inclusion, and per eye for that eye's point to
            be shown. A scalar applies the same threshold to all three; a dict
            ``{'distance': .., 'left': .., 'right': ..}`` sets them independently
            (a missing key skips the threshold for that condition).
        fit : bool
            Overlay a linear fit per eye (red=left, blue=right), each over the
            cells reliable in that eye, with slope + Pearson r in the legend.
        stim_device : goggles device, optional
            The session whose pixels are on the y axis (i.e. the **monocular**
            device). When given, overlays each eye's assumed deg->pixel fit and
            the band of pixels that reached that eye's screen, so the recovered
            fit can be read against the assumed one. See ``_draw_stim_reference``.
        """
        left_pts = self._calibration_subset('left', cells=cells, corr_threshold=corr_threshold)
        right_pts = self._calibration_subset('right', cells=cells, corr_threshold=corr_threshold)
        both_pts = pd.merge(left_pts, right_pts, on='cell', suffixes=('_left', '_right'))
        n_cells = len(set(left_pts['cell']) | set(right_pts['cell']))

        fig, ax = plt.subplots(figsize=(6, 6))

        bands = []
        if stim_device is not None:
            for eye, color, pts in (('left', 'red', left_pts), ('right', 'blue', right_pts)):
                bands.append(self._draw_stim_reference(ax, eye, stim_device, color, sub=pts))
        # vertical connectors for binocular cells first, so the points draw on top
        for _, row in both_pts.iterrows():
            ax.plot([row['distance_RF_left'], row['distance_RF_left']],
                    [row['px_left'], row['px_right']],
                    color='gray', linewidth=1, alpha=0.6, zorder=1)
        ax.scatter(left_pts['distance_RF'], left_pts['px'], color='red', edgecolor='black',
                   alpha=0.8, zorder=2, label=f'Left eye (n={len(left_pts)})')
        ax.scatter(right_pts['distance_RF'], right_pts['px'], color='blue', edgecolor='black',
                   alpha=0.8, zorder=2, label=f'Right eye (n={len(right_pts)})')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        if fit:
            for eye, color, name in (('left', 'red', 'Left'), ('right', 'blue', 'Right')):
                f = self.fit_eye_calibration(eye, cells=cells, corr_threshold=corr_threshold)
                if f['n'] >= 2:
                    xs = np.linspace(f['deg_min'], f['deg_max'], 100)
                    ax.plot(xs, f['slope'] * xs + f['intercept'], color=color, linewidth=2, zorder=3,
                            label=f"{name}: px = {f['slope']:.2f}·deg {f['intercept']:+.1f}  "
                                  f"(r={f['r']:.2f})")

        if cells is not None:   # only label points for a hand-picked group
            labelled = pd.concat([right_pts, left_pts]).drop_duplicates(subset='cell')
            for _, row in labelled.iterrows():
                ax.annotate(str(int(row['cell'])), (row['distance_RF'], row['px']),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 2), textcoords='offset points')

        # mean vertical distance between the two eyes' RFs (binocular cells only)
        vdist = (both_pts['px_right'] - both_pts['px_left']).to_numpy(float)
        if len(vdist):
            mean_v, std_v = float(np.mean(vdist)), float(np.std(vdist))
            ax.text(0.02, 0.98,
                    f'mean |R−L| = {mean_v:.1f} ± {std_v:.1f} px (n={len(vdist)} binoc.)',
                    transform=ax.transAxes, ha='left', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

        rho_txt = self._rho_txt(corr_threshold)
        base = 'Hand-picked cells' if cells is not None else 'Reliable in distance & ≥1 eye'
        ax.set_xlabel('Distance RF (deg)')
        ax.set_ylabel('Goggles RF (px)')
        ax.set_title(f'{base}{rho_txt} (n={n_cells})')
        ax.legend(frameon=False, fontsize=10 if stim_device is not None else 12)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(self._expand_ylim(ylim, bands))
        plt.tight_layout()
        return fig

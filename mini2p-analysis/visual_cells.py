"""Classify "visual" cells: reliable egocentric map in light, NOT in dark.

A cell is **visual** when its egocentric boundary map is significantly *reliable*
(reproducible across time) in the LIGHT but **not** in the DARK:

    is_visual = (light reliability > light shuffle-null p) AND
                (dark  reliability <= dark  shuffle-null p)

Reliability is the mean split-half correlation of the occupancy-normalized ego map across
balanced partitions of the session's natural blocks. Significance is judged *per condition*
against a circular-time-shift null -- the cell's activity rolled out of register with the
wall geometry, which preserves each signal's temporal structure but destroys their pairing
-- the reliability convention the project already uses (``disparity-analysis``
``CellAnalysis``, legacy ``DistanceProject``).

Making "doesn't fire in the dark" an *absolute* bar (dark no more reliable than chance)
rejects cells that code boundaries with the lights off too (reliable in both conditions),
and removes the light-vs-dark firing-rate confound (more spikes -> cleaner map) that makes a
plain light>dark comparison fire on almost everything.

Session structure: 10 trials x ``[1350 light | 1350 dark]`` frames -> 10 natural blocks per
condition; with ``n_chunks=10`` the ``C(10,5)/2 = 126`` balanced splits are enumerated
exactly.
"""

from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from cell_explorer import CellExplorer

COND_LIGHTVAL = {"light": 0, "dark": 1}


def _corr_lastaxis(a, b):
    """Pearson correlation over the last axis of two equally-shaped arrays; NaN/degenerate
    -> 0 (matching the project's ``_safe_corr``). Broadcasts over leading axes, so it scores
    every (split, cell) pair at once."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum(axis=-1))
    return np.where(np.isfinite(r), r, 0.0)


class VisualClassifier:
    """Light-reliable / dark-unreliable egocentric-map classifier for one session."""

    def __init__(self, source, n_chunks=10, activity="spk_rate_thr"):
        """
        Parameters
        ----------
        source : CellExplorer | str
            Existing ``CellExplorer`` (reuses its loaded arrays) or a path to an
            ``ebr_pipeline_*.pkl``.
        n_chunks : int
            Chunks per condition (default 10 = the natural blocks; higher sub-divides them).
        activity : str
            Per-frame activity column for the rate maps (default ``spk_rate_thr``).
        """
        # detect by path-ness, not isinstance(CellExplorer), so a reloaded class can't break it
        ce = CellExplorer(source) if isinstance(source, (str, os.PathLike)) else source
        self.ce = ce
        self.settings = ce.settings
        self.session_id = ce.session_id

        s = ce.settings
        self.n_rays = 360 // s["ego_rays_deg"]
        self.n_dist = round(s["ego_radius"] / s["ego_bin_size"])
        self.bin_size = s["ego_bin_size"]
        self.radius = s["ego_radius"]
        self.n_cells = ce.n_cells
        self.n_chunks = n_chunks

        self.ray_dist = np.stack(ce.df["ray_distances"].values).astype(float)  # (F, n_rays)
        self.activity = ce.spk_mag if activity == "spk_rate_thr" else np.stack(
            [np.asarray(v, dtype=float) for v in ce.df[activity].values]
        )                                                                       # (F, n_cells)
        self.spk = ce.spk                                                       # (F, n_cells) bool
        self.light_col = ce.light_col

        self._bin_cache = {}   # condition -> (M, occ, frames)
        self._last = None      # cached classify() results for plotting

    # --------------------------------------------------------------- binning
    def _build_binning(self, condition):
        """Sparse frame->(chunk,ray,dist) map for one condition, built once.

        Returns ``(M, occ, frames)`` where ``M`` is a CSR matrix
        ``(n_chunks·n_rays·n_dist, n_frames)`` (1 per valid frame/ray), ``occ`` the fixed
        ``(n_chunks, n_rays, n_dist)`` occupancy, and ``frames`` the condition's frame
        indices. Any per-frame activity vector maps to per-chunk spike maps via ``M @ act``,
        so a circularly-shifted activity re-bins with a single sparse matmul.
        """
        if condition in self._bin_cache:
            return self._bin_cache[condition]

        frames = np.where(self.light_col == COND_LIGHTVAL[condition])[0]
        rd = self.ray_dist[frames]                                    # (Fc, n_rays)
        Fc = len(frames)
        chunk_id = np.empty(Fc, dtype=int)
        for ci, idx in enumerate(np.array_split(np.arange(Fc), self.n_chunks)):
            chunk_id[idx] = ci

        valid = (rd >= 0) & (rd < self.radius)
        binr = np.floor(np.where(valid, rd, -1.0) / self.bin_size).astype(int)
        valid &= binr < self.n_dist

        nrd = self.n_rays * self.n_dist
        rows, cols = [], []
        for r in range(self.n_rays):
            fi = np.where(valid[:, r])[0]
            rows.append(chunk_id[fi] * nrd + r * self.n_dist + binr[fi, r])
            cols.append(fi)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(self.n_chunks * nrd, Fc))
        occ = np.asarray(M.sum(axis=1)).reshape(
            self.n_chunks, self.n_rays, self.n_dist).astype(np.float32)
        self._bin_cache[condition] = (M, occ, frames)
        return M, occ, frames

    def _spk_from_activity(self, M, act):
        """Per-chunk spike maps ``(n_chunks, n_cells, n_rays, n_dist)`` for an activity
        array ``act`` ``(Fc, n_cells)`` via one sparse matmul."""
        flat = M @ act                                                # (n_chunks·n_rays·n_dist, n_cells)
        return flat.reshape(self.n_chunks, self.n_rays, self.n_dist,
                            self.n_cells).transpose(0, 3, 1, 2)

    # ------------------------------------------------------------- splits
    def _make_splits(self, n_splits, rng):
        """Balanced chunk-membership matrix ``S`` ``(n_splits, n_chunks)`` (row = 1 for the
        chunks in half A, 0 fixed in A so each unordered split appears once). Enumerate all
        ``C(n_chunks-1, half-1)`` splits when feasible, else randomly sample ``n_splits``."""
        K, half = self.n_chunks, self.n_chunks // 2
        total = comb(K - 1, half - 1)
        if n_splits is None or n_splits >= total:
            sets = [(0,) + c for c in combinations(range(1, K), half - 1)]
        else:
            seen = set()
            while len(seen) < n_splits:
                c = tuple(sorted(rng.choice(range(1, K), half - 1, replace=False).tolist()))
                seen.add((0,) + c)
            sets = list(seen)
        S = np.zeros((len(sets), K), dtype=np.float32)
        for i, A in enumerate(sets):
            S[i, list(A)] = 1.0
        return S, total

    def _split_corrs(self, spk, occ, S, fmask):
        """``(n_splits, n_cells)`` split-half correlations, fully vectorized.

        Each half spans ~half the session so every visited ego bin is covered in both
        halves; correlating over the fixed ``fmask`` (bins ever visited) is therefore exact
        and lets all splits be scored with two matmuls instead of a Python loop.
        """
        K, nC, B = self.n_chunks, self.n_cells, self.n_rays * self.n_dist
        spk3 = spk.reshape(K, nC, B)                       # (K, nC, B)
        occ2 = occ.reshape(K, B)                           # (K, B)
        spkA = np.tensordot(S, spk3, axes=(1, 0))          # (n_splits, nC, B)
        occA = S @ occ2                                    # (n_splits, B)
        spkB = spk3.sum(0)[None] - spkA
        occB = occ2.sum(0)[None] - occA
        with np.errstate(divide="ignore", invalid="ignore"):
            mapA = np.where(occA[:, None, :] > 0, spkA / occA[:, None, :], 0.0)
            mapB = np.where(occB[:, None, :] > 0, spkB / occB[:, None, :], 0.0)
        if not fmask.all():                                # halves are ~half the session,
            mapA, mapB = mapA[:, :, fmask], mapB[:, :, fmask]  # so this is usually a no-op
        return _corr_lastaxis(mapA, mapB)

    def _reliability(self, condition, S, n_shuffles, rng):
        """``(real, null)``: real reliability ``(n_cells,)`` = mean split-half corr, and
        ``null`` ``(n_shuffles, n_cells)`` from circularly time-shifted activity."""
        M, occ, frames = self._build_binning(condition)
        fmask = occ.sum(0).reshape(-1) > 0                 # ego bins ever visited
        act = self.activity[frames].astype(np.float32)
        real = self._split_corrs(self._spk_from_activity(M, act), occ, S, fmask).mean(0)

        Fc = len(frames)
        lo = max(50, Fc // 20)                             # avoid trivially small shifts
        null = np.empty((n_shuffles, self.n_cells))
        for s in range(n_shuffles):
            shifted = np.roll(act, int(rng.integers(lo, Fc - lo)), axis=0)
            null[s] = self._split_corrs(
                self._spk_from_activity(M, shifted), occ, S, fmask).mean(0)
        return real, null

    # ------------------------------------------------------------- classify
    def classify(self, percentile=99, min_light_corr=0.5,
                 n_shuffles=100, n_splits=40, seed=0, verbose=True):
        """Per-cell ``DataFrame`` with the visual-cell verdict.

        A condition is *reliable* when its real split-half reliability exceeds the
        ``percentile`` of its circular-shift null. ``is_visual`` = light reliable AND its
        light reliability >= ``min_light_corr`` (absolute floor) AND dark NOT reliable.

        ``n_splits`` balanced splits are sampled for the reliability mean (default 40 — the
        mean is stable well below the full ``C(n_chunks-1, half-1)``; pass ``None`` to
        enumerate every split). ``n_shuffles`` circular shifts build the per-condition null.
        Both drive runtime. ``verbose`` prints a one-block session summary (also available
        any time via :meth:`summary`).
        """
        rng = np.random.default_rng(seed)
        S, total = self._make_splits(n_splits, rng)
        res = {}
        for cond in ("light", "dark"):
            real, null = self._reliability(cond, S, n_shuffles, rng)
            res[cond] = {"real": real, "null": null,
                         "p": np.percentile(null, percentile, axis=0)}

        n_spk_light = self.spk[self.light_col == 0].sum(axis=0)
        n_spk_dark = self.spk[self.light_col == 1].sum(axis=0)
        light_sig = res["light"]["real"] > res["light"]["p"]
        dark_sig = res["dark"]["real"] > res["dark"]["p"]
        is_visual = light_sig & ~dark_sig & (res["light"]["real"] >= min_light_corr)

        df = pd.DataFrame({
            "cell": np.arange(self.n_cells),
            "light_reliability": res["light"]["real"], "light_null_p": res["light"]["p"],
            "light_sig": light_sig,
            "dark_reliability": res["dark"]["real"], "dark_null_p": res["dark"]["p"],
            "dark_sig": dark_sig,
            "is_visual": is_visual,
            "n_spk_light": n_spk_light, "n_spk_dark": n_spk_dark,
        })
        self._last = {"res": res, "percentile": percentile, "df": df,
                      "min_light_corr": min_light_corr,
                      "n_splits": S.shape[0], "total_splits": total, "n_shuffles": n_shuffles}
        if verbose:
            print(self.summary())
        return df

    def summary(self):
        """One-block text report of the most recent ``classify()`` run."""
        if self._last is None:
            return "no classification yet -- call classify()"
        L = self._last
        df, p, floor = L["df"], L["percentile"], L["min_light_corr"]
        n = len(df)
        light_sig = df["light_sig"].to_numpy()
        dark_sig = df["dark_sig"].to_numpy()
        vis = df["is_visual"].to_numpy()
        also_dark = int((light_sig & dark_sig).sum())
        below_floor = int((light_sig & ~dark_sig & (df["light_reliability"] < floor)).sum())
        return "\n".join([
            f"[VisualClassifier] {self.session_id} -- {n} cells",
            f"  params: percentile={p}  min_light_corr={floor:g}  "
            f"n_splits={L['n_splits']}/{L['total_splits']}  n_shuffles={L['n_shuffles']}",
            f"  light reliable (real > p{p} null): {int(light_sig.sum())}   "
            f"dark reliable: {int(dark_sig.sum())}",
            f"  VISUAL (light-reliable & >={floor:g} & dark-unreliable): "
            f"{int(vis.sum())}  ({100 * vis.mean():.1f}%)",
            f"  rejected of {int(light_sig.sum())} light-reliable: "
            f"{also_dark} also dark-reliable, {below_floor} below floor {floor:g}",
            f"  median reliability (real / p{p} null):  "
            f"light {df['light_reliability'].median():.2f} / {df['light_null_p'].median():.2f}   "
            f"dark {df['dark_reliability'].median():.2f} / {df['dark_null_p'].median():.2f}",
        ])

    # ---------------------------------------------------------------- plot
    def plot_cell(self, cell, figsize=(15, 4.4), ego_cmap="Blues", bins=20):
        """light ego map | dark ego map | reliability vs shuffle null (real lines + null
        histograms + threshold), with the verdict. Returns the figure."""
        if self._last is None:
            self.classify()
        res, p = self._last["res"], self._last["percentile"]
        row = self._last["df"].iloc[cell]

        ego_light = self.ce._ego_maps("light")[cell]
        ego_dark = self.ce._ego_maps("dark")[cell]
        finite = np.concatenate([ego_light[np.isfinite(ego_light)],
                                 ego_dark[np.isfinite(ego_dark)]])
        vmax = np.percentile(finite, 95) if finite.size else None

        fig = plt.figure(figsize=figsize)
        axL = fig.add_subplot(1, 3, 1, projection="polar")
        axD = fig.add_subplot(1, 3, 2, projection="polar")
        axH = fig.add_subplot(1, 3, 3)
        CellExplorer._draw_ego(axL, ego_light, ego_cmap, "light ego map", vmin=0, vmax=vmax)
        CellExplorer._draw_ego(axD, ego_dark, ego_cmap, "dark ego map", vmin=0, vmax=vmax)

        for cond, color in (("light", "goldenrod"), ("dark", "navy")):
            axH.hist(res[cond]["null"][:, cell], bins=bins, color=color, alpha=0.35,
                     density=True, label=f"{cond} null")
            axH.axvline(res[cond]["real"][cell], color=color, lw=2.5, label=f"{cond} real")
            axH.axvline(res[cond]["p"][cell], color=color, ls="--", lw=1.2)
        axH.set_xlabel("split-half reliability"); axH.set_ylabel("null density")
        axH.legend(fontsize=8)
        verdict = "VISUAL" if bool(row["is_visual"]) else "not visual"
        flags = f"light_sig={bool(row['light_sig'])}  dark_sig={bool(row['dark_sig'])}"
        axH.set_title(f"{verdict}  ({flags})")

        fig.suptitle(f"{self.session_id} - cell {cell}  "
                     f"(light {row['light_reliability']:.2f}/p{p} {row['light_null_p']:.2f} | "
                     f"dark {row['dark_reliability']:.2f}/p{p} {row['dark_null_p']:.2f})")
        fig.tight_layout()
        return fig

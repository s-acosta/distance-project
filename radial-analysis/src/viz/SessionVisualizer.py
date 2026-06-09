import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from src.utils import setup_plot_style, create_new_result_dir
from scipy.stats import binned_statistic_2d, pearsonr, mannwhitneyu, fisher_exact
from itertools import combinations
from src.viz.CellVisualizer import CellVisualizer
from matplotlib.patches import Polygon


class SessionVisualizerBase():
    """
    Base class with shared visualization functionality for session DataFrames.

    Provides filtering, feature extraction, and plotting methods common to both
    single-session and multi-session analysis. Not meant to be instantiated directly.

    Use SingleSessionVisualizer for single-session data (Suite2P coordinates)
    or MultiSessionVisualizer for multi-session data (widefield coordinates).

    Parameters
    ----------
    session_df : pd.DataFrame
        Session DataFrame with cell statistics. Must contain 'session_name' column.
    base_data_dir : str, optional
        Base directory for processed session data.
    """

    _AXIS_LIMITS = {
        'RF': (-20, 60),
        'azi': (-20, 90),
        'alt': (-50, 50),
    }

    def __init__(self, session_df,
                 base_data_dir=r'D:\data-distance-coding-project\processed-data'):

        self.session_df = session_df
        self.base_data_dir = base_data_dir

        self.distance_colors = setup_plot_style()
        self.modulation_colors = {
            'monocular': 'coral',
            'binocular': 'lightblue',
            'both': 'purple',
            'none': 'gray'
        }

        self.modulated_classes = {
            'monocular': self.filter_by_modulation(self.session_df, modulation_type='monocular'),
            'binocular': self.filter_by_modulation(self.session_df, modulation_type='binocular'),
            'both':      self.filter_by_modulation(self.session_df, modulation_type='both'),
            'none':      self.filter_by_modulation(self.session_df, modulation_type='none'),
        }
        self.modulated_classes[None] = self.filter_by_reliability(self.session_df, reliable_type='reliable')

        self.reliable_classes = {
            'reliable':     self.filter_by_reliability(self.session_df, reliable_type='reliable'),
            'not_reliable': self.filter_by_reliability(self.session_df, reliable_type='not_reliable'),
        }

    # -------------------------------------------------------------------------
    # Abstract hooks — subclasses must implement these
    # -------------------------------------------------------------------------

    def _get_background_image(self):
        raise NotImplementedError("Subclasses must implement _get_background_image()")

    def _get_background_extent(self):
        raise NotImplementedError("Subclasses must implement _get_background_extent()")

    def _get_cell_coordinates(self, input_df):
        raise NotImplementedError("Subclasses must implement _get_cell_coordinates()")

    def _get_session_title(self):
        raise NotImplementedError("Subclasses must implement _get_session_title()")

    # -------------------------------------------------------------------------
    # Default hook implementations — subclasses may override
    # -------------------------------------------------------------------------

    def _apply_axis_adjustments(self, ax):
        """Apply mode-specific axis adjustments after plotting. Default: no-op."""
        pass

    def _overlay_area_boundaries(self, ax, show_legend=True):
        """Overlay visual area boundaries on a plot. Default: no-op."""
        pass

    def _get_area_colors(self, group_by):
        """Return area-name → color mapping for bar charts. Default: empty dict."""
        return {}

    @property
    def _invert_axes(self):
        """Whether axes should be inverted (True for single-session). Default: False."""
        return False

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_available_sessions(self):
        """
        Get list of all unique session names in the DataFrame.

        Returns
        -------
        list
            List of session name strings.
        """
        return self.session_df['session_name'].unique().tolist()

    def get_available_areas(self):
        """
        Get list of all available visual area names.

        Only available in MultiSessionVisualizer.

        Raises
        ------
        ValueError
            Always raises when called on SingleSessionVisualizer.
        """
        raise ValueError(
            "get_available_areas is only available in MultiSessionVisualizer."
        )

    # -------------------------------------------------------------------------
    # Filter methods
    # -------------------------------------------------------------------------

    def filter_by_reliability(self, input_df, reliable_type='reliable', threshold=0.85):

        has_monocular = input_df['monocular'].notna().any()
        has_binocular = input_df['binocular'].notna().any()

        if reliable_type == 'reliable':
            mask = pd.Series(True, index=input_df.index)
            if has_monocular:
                mask &= input_df['monocular'].apply(lambda x: x['reliable']) == True
            if has_binocular:
                mask &= input_df['binocular'].apply(lambda x: x['reliable']) == True
            corr_mask = pd.Series(False, index=input_df.index)
            if has_monocular:
                corr_mask |= input_df['monocular'].apply(lambda x: x['reliable_corr']) >= threshold
            if has_binocular:
                corr_mask |= input_df['binocular'].apply(lambda x: x['reliable_corr']) >= threshold
            output_df = input_df[mask & corr_mask]

        elif reliable_type == 'not_reliable':
            mask = pd.Series(True, index=input_df.index)
            corr_mask = pd.Series(True, index=input_df.index)
            if has_monocular:
                mask &= input_df['monocular'].apply(lambda x: x['reliable']) == False
                corr_mask &= input_df['monocular'].apply(lambda x: x['reliable_corr']) < threshold
            if has_binocular:
                mask &= input_df['binocular'].apply(lambda x: x['reliable']) == False
                corr_mask &= input_df['binocular'].apply(lambda x: x['reliable_corr']) < threshold
            output_df = input_df[mask & corr_mask]

        return output_df

    def filter_by_modulation(self, input_df, modulation_type='monocular'):

        reliable_df = self.filter_by_reliability(input_df, reliable_type='reliable', threshold=0.85)

        has_monocular = input_df['monocular'].notna().any()
        has_binocular = input_df['binocular'].notna().any()

        if modulation_type == 'monocular':
            if not has_monocular:
                return reliable_df.iloc[0:0]
            if has_binocular:
                output_df = reliable_df[
                    (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
                    (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05)
                ]
            else:
                output_df = reliable_df[
                    reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05
                ]

        elif modulation_type == 'binocular':
            if not has_binocular:
                return reliable_df.iloc[0:0]
            if has_monocular:
                output_df = reliable_df[
                    (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05) &
                    (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
                ]
            else:
                output_df = reliable_df[
                    reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05
                ]

        elif modulation_type == 'both':
            if not has_monocular or not has_binocular:
                return reliable_df.iloc[0:0]
            output_df = reliable_df[
                (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
                (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
            ]

        elif modulation_type == 'none':
            mask = pd.Series(True, index=reliable_df.index)
            if has_monocular:
                mask &= reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05
            if has_binocular:
                mask &= reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05
            output_df = reliable_df[mask]

        else:
            raise ValueError(f"Unknown modulation_type '{modulation_type}'. "
                             f"Must be one of: 'monocular', 'binocular', 'both', 'none'.")

        return output_df

    def filter_by_area(self, input_df, area_name='V1'):

        output_df = input_df[input_df['area'] == area_name]

        return output_df

    def filter_by_session(self, input_df, session_name='SAM041_251128_001'):

        output_df = input_df[input_df['session_name'] == session_name]

        return output_df

    # -------------------------------------------------------------------------
    # Feature extraction methods
    # -------------------------------------------------------------------------

    def extract_condition_feature(self, input_df, condition=None, feature='RF'):

        if condition == 'monocular' or condition == 'binocular':
            feature_values = input_df[condition].apply(lambda x: x[feature]).values

        elif condition == 'both' or condition == 'none':
            monocular_values = input_df['monocular'].apply(lambda x: x[feature]).values
            binocular_values = input_df['binocular'].apply(lambda x: x[feature]).values
            feature_values = (monocular_values + binocular_values) / 2

        return feature_values

    def extract_general_feature(self, input_df, feature='RF'):

        feature_values = input_df[feature].values
        return feature_values

    def _filter_by_rf_range(self, input_df, condition, rf_range):
        if rf_range is None or len(input_df) == 0:
            return input_df
        rf_min, rf_max = rf_range
        if condition in ('reliable', 'un-modulated'):
            rf_vals = self.extract_general_feature(input_df, feature='RF')
        else:
            rf_vals = self.extract_condition_feature(input_df, condition=condition, feature='RF')
        rf_vals = np.asarray(rf_vals, dtype=float)
        mask = (rf_vals >= rf_min) & (rf_vals <= rf_max)
        return input_df[mask]

    def _get_x_axis_values(self, target_df, x_axis, condition, scope_df=None):
        """
        Per-cell x-axis values. RF is returned as-is in degrees (condition-aware).
        'azi' / 'alt' are linearly rescaled from `scope_df`'s min/max into the
        canonical range from `_AXIS_LIMITS`, so spatial relationships are preserved.
        """
        if x_axis == 'RF':
            if condition in ('reliable', 'un-modulated'):
                return np.asarray(self.extract_general_feature(target_df, feature='RF'), dtype=float)
            return np.asarray(self.extract_condition_feature(target_df, condition=condition, feature='RF'), dtype=float)

        if x_axis not in ('azi', 'alt'):
            raise ValueError(f"x_axis must be 'RF', 'azi', or 'alt', got {x_axis!r}")

        if scope_df is None:
            scope_df = target_df
        target_vals = np.asarray(target_df[x_axis].values, dtype=float)
        scope_vals = np.asarray(scope_df[x_axis].values, dtype=float)
        source_min = np.nanmin(scope_vals)
        source_max = np.nanmax(scope_vals)
        target_min, target_max = self._AXIS_LIMITS[x_axis]
        if source_max > source_min:
            return ((target_vals - source_min) / (source_max - source_min)
                    * (target_max - target_min) + target_min)
        return target_vals

    # -------------------------------------------------------------------------
    # Shared plotting methods
    # -------------------------------------------------------------------------

    def plot_responsive_bars(self, group_by=None):

        # Get unique areas
        if group_by == 'area':
            groups = self.get_available_areas()
        elif group_by == 'session':
            groups = self.get_available_sessions()
        elif group_by is None:
            groups = ['all_cells']
        else:
            raise ValueError(f"group_by must be 'area' or 'session', got {group_by}")

        fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.5), 5))
        # Calculate responsive cells for each group
        group_data = {}
        for group in groups:

            if group_by == 'area':
                group_df = self.filter_by_area(self.session_df, area_name=str(group))
            elif group_by == 'session':
                group_df = self.filter_by_session(self.session_df, session_name=str(group))
            elif group_by is None:
                group_df = self.session_df

            n_reliable_cells = len(self.filter_by_reliability(group_df, reliable_type='reliable', threshold=0.85))
            total_cells = len(group_df)

            group_data[group] = {
                'total': total_cells,
                'reliable': n_reliable_cells
            }

        # Build area color map via subclass hook
        area_colors = self._get_area_colors(group_by)

        bar_width = 0.75
        for i, group in enumerate(groups):
            total = group_data[group]['total']
            reliable = group_data[group]['reliable']
            pct = reliable / total * 100 if total > 0 else 0

            bar_color = area_colors.get(group, 'lightgreen')

            # Plot total cells as grey bar
            ax.bar(group, total, color='lightgray', width=bar_width, label='Total Cells' if i == 0 else "")
            # Plot reliable cells with area color
            ax.bar(group, reliable, color=bar_color, width=bar_width, alpha=0.7, label='Reliable Cells' if i == 0 else "")

            # Add percentage text label on top of total bar
            ax.text(group, total, f'{pct:.1f}%\n({reliable}/{total})',
                   ha='center', va='bottom', fontsize=13)

        ax.set_ylabel('Number of Cells', fontsize=12)

        if group_by is 'session':
            # Split session names by underscore and show only 2nd and 3rd components
            groups = ['\n'.join(group.split('_')[3]) for group in groups]

        ax.set_xlabel(group_by.capitalize() if group_by is not None else 'All Cells', fontsize=12)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend()

        # add title with less padding so they appear closer
        fig.suptitle('Responsive Cells', fontsize=18, color='black', y=1.05)
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', y=1.05)

        # plt.tight_layout()

        return fig

    def plot_modulation_bars(self, group_by=None):

        # Get unique groups
        if group_by == 'area':
            groups = self.get_available_areas()
        elif group_by == 'session':
            groups = self.get_available_sessions()
        elif group_by is None:
            groups = ['all_cells']
        else:
            raise ValueError(f"group_by must be 'area' or 'session', got {group_by}")

        modulation_types = ['monocular', 'binocular', 'both']
        n_modulations = len(modulation_types)

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 7))

        # Calculate modulation counts for each group
        group_data = {}
        for group in groups:
            if group_by == 'area':
                group_df = self.filter_by_area(self.session_df, area_name=str(group))
            elif group_by == 'session':
                group_df = self.filter_by_session(self.session_df, session_name=str(group))
            elif group_by is None:
                group_df = self.session_df

            total_cells = len(group_df)
            reliable_count = len(self.filter_by_reliability(group_df, reliable_type='reliable', threshold=0.85))

            modulation_counts = {}
            for mod_type in modulation_types:
                modulation_counts[mod_type] = len(self.filter_by_modulation(group_df, modulation_type=mod_type))

            group_data[group] = {
                'total': total_cells,
                'reliable': reliable_count,
                'modulations': modulation_counts
            }

        # Filter out groups with less than 10% reliable cells
        groups = [g for g in groups if group_data[g]['total'] > 0 and
                  group_data[g]['reliable'] / group_data[g]['total'] >= 0.10]

        # Set up bar positions with separation between groups
        x = np.arange(len(groups)) * 1.2
        bar_width = 0.28
        bar_gap = 0.03

        # Plot bars for each modulation type
        for j, mod_type in enumerate(modulation_types):
            offset = (j - n_modulations/2 + 0.5) * (bar_width + bar_gap)

            for i, group in enumerate(groups):
                reliable = group_data[group]['reliable']
                mod_count = group_data[group]['modulations'][mod_type]
                pct = mod_count / reliable * 100 if reliable > 0 else 0

                # Plot modulated cells as colored bar (percentage) with outline
                ax.bar(x[i] + offset, pct, color=self.modulation_colors[mod_type], width=bar_width,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=mod_type.capitalize() if i == 0 else "")

                # Add n cells as text label
                ax.text(x[i] + offset, pct + 1, f'n={mod_count}',
                       ha='center', va='bottom', fontsize=12, fontweight='medium')

        ax.set_ylabel('Percentage of Reliable Cells (%)', fontsize=16)
        if group_by is 'session':
            # Split session names by underscore and show only 2nd and 3rd components
            groups = ['\n'.join(group.split('_')[3]) for group in groups]

        ax.set_xlabel(group_by.capitalize() if group_by is not None else '', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add subtle horizontal grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        # add title with less padding so they appear closer
        plt.tight_layout()
        fig.suptitle('Modulated Cells', fontsize=22, color='black', y=1.1, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=20, color='grey', pad=2.5, y=1.05)

        return fig

    def plot_modulation_index(self):

        # Create a boxplot of modulation strength for monocular, binocular, both conditions, and non-modulated
        n_conditions = len(self.modulated_classes.keys())
        fig, ax = plt.subplots(figsize=(2*n_conditions, 6))

        # Prepare data for boxplot
        data_for_boxplot = []
        box_labels = []

        conditions = list(self.modulated_classes.keys())
        cells_to_plot = [self.modulated_classes[condition] for condition in conditions]

        # for every condition in modulated_cells, get the modulation strength
        for i, condition in enumerate(conditions):
            print(condition)
            DTI_condition = self.extract_condition_feature(cell=cells_to_plot[i], condition=condition, feature='DTI')

            data_for_boxplot.append(np.array(DTI_condition))
            box_labels.append(condition)

        bp = ax.boxplot(data_for_boxplot, patch_artist=True, labels=box_labels, showfliers=False)

        # Customize boxplot appearance
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=self.modulation_colors[conditions[i]], alpha=0.7)

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        # Add individual points with jitter
        for i, data in enumerate(data_for_boxplot):
            # Create jitter effect
            x = np.random.normal(i+1, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.7, s=40, color=self.modulation_colors[conditions[i]], edgecolor='black')

        # Add labels and title
        ax.set_xlabel('Condition', fontsize=16)
        ax.set_xticklabels(box_labels, fontsize=12)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Modulation Strength')

        # add title with less padding so they appear closer
        fig.suptitle('Modulation Index by Condition', fontsize=18, color='black', y=0.94)
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show plot
        plt.tight_layout()

        return fig

    def plot_distance_bars(self, group_by=None, condition='both', min_cells=10, rf_range=None):

        # Get unique groups
        if group_by == 'area':
            groups = self.get_available_areas()
        elif group_by == 'session':
            groups = self.get_available_sessions()
        elif group_by is None:
            groups = ['all_cells']
        else:
            raise ValueError(f"group_by must be 'area' or 'session', got {group_by}")

        distances = [0, 1, 2, 3]
        n_distances = len(distances)

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 7))

        # Calculate distance counts for each group
        group_data = {}
        for group in groups:
            if group_by == 'area':
                group_df = self.filter_by_area(self.session_df, area_name=str(group))
            elif group_by == 'session':
                group_df = self.filter_by_session(self.session_df, session_name=str(group))
            elif group_by is None:
                group_df = self.session_df

            total_cells = len(group_df)
            reliable_df = self.filter_by_reliability(group_df, reliable_type='reliable', threshold=0.85)
            reliable_count = len(reliable_df)

            if condition in ('reliable', 'un-modulated'):
                if condition == 'reliable':
                    target_df = reliable_df
                else:
                    target_df = self.filter_by_modulation(group_df, modulation_type='none')
                target_df = self._filter_by_rf_range(target_df, condition, rf_range)
                target_count = len(target_df)
                if target_count > 0:
                    pref_dist_values = target_df['distance'].values
                    pref_distances = [int(np.round(d)) for d in pref_dist_values]
                    distance_counts = {d: np.sum(np.array(pref_distances) == d) for d in distances}
                else:
                    distance_counts = {d: 0 for d in distances}
            else:
                modulated_df = self.filter_by_modulation(group_df, modulation_type=condition)
                modulated_df = self._filter_by_rf_range(modulated_df, condition, rf_range)
                target_count = len(modulated_df)

                if target_count > 0:
                    if condition == 'monocular':
                        pref_dist_values = modulated_df['monocular'].apply(lambda x: x['distance']).values
                    elif condition == 'binocular':
                        pref_dist_values = modulated_df['binocular'].apply(lambda x: x['distance']).values
                    else:  # 'both'
                        monocular_dist = modulated_df['monocular'].apply(lambda x: x['distance']).values
                        binocular_dist = modulated_df['binocular'].apply(lambda x: x['distance']).values
                        pref_dist_values = (monocular_dist + binocular_dist) / 2
                    pref_distances = [int(np.round(d)) for d in pref_dist_values]
                    distance_counts = {d: np.sum(np.array(pref_distances) == d) for d in distances}
                else:
                    distance_counts = {d: 0 for d in distances}

            group_data[group] = {
                'total': total_cells,
                'reliable': reliable_count,
                'target': target_count,
                'distances': distance_counts
            }

        # Filter out groups with insufficient cells
        groups = [g for g in groups if group_data[g]['total'] > 0 and
                  group_data[g]['reliable'] / group_data[g]['total'] >= 0.10 and
                  group_data[g]['target'] >= min_cells]

        # Set up bar positions with separation between groups
        x = np.arange(len(groups)) * 1.2
        bar_width = 0.20
        bar_gap = 0.03

        # Plot bars for each distance
        for j, dist in enumerate(distances):
            offset = (j - n_distances/2 + 0.5) * (bar_width + bar_gap)

            for i, group in enumerate(groups):
                target = group_data[group]['target']
                dist_count = group_data[group]['distances'][dist]
                pct = dist_count / target * 100 if target > 0 else 0

                ax.bar(x[i] + offset, pct, color=self.distance_colors[j], width=bar_width,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=f'Distance {dist}' if i == 0 else "")

                ax.text(x[i] + offset, pct + 1, f'n={dist_count}',
                       ha='center', va='bottom', fontsize=12, fontweight='medium')

        if condition in ('reliable', 'un-modulated'):
            ax.set_ylabel('Percentage of Reliable Cells (%)', fontsize=16)
        else:
            ax.set_ylabel('Percentage of Modulated Cells (%)', fontsize=16)
        ax.set_xlabel(group_by.capitalize() if group_by is not None else '', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add subtle horizontal grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        condition_label = {'reliable': 'Reliable', 'un-modulated': 'Un-modulated', 'both': 'Combined'}.get(condition, condition.capitalize())
        fig.suptitle(f'Preferred Distance ({condition_label})', fontsize=22, color='black', y=1, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=20, color='grey', y=1.05)

        plt.tight_layout()

        return fig

    def plot_preferred_distance_polar(self, condition='binocular', ax=None):

        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1, polar=True)
        else:
            fig = ax.get_figure()

        ax.set_theta_zero_location('N')
        # Add gray background for central region (-25 to 25 degrees)
        ax.axvspan(np.radians(-25), np.radians(25), alpha=0.5, color='gray', zorder=0)
        # Add colored y-spans for distance ranges
        for dist in range(1, 5):
            ax.axhspan(dist - 0.30, dist + 0.30, alpha=0.4, color=self.distance_colors[dist-1], zorder=0)

        cells_to_plot = self.modulated_classes[condition]

        if len(cells_to_plot) == 0:
            ax.set_title(f'{condition.capitalize() if condition else "All"} (no cells)', fontsize=18, pad=2)
            return fig, ax

        pref_distances = self.extract_condition_feature(cells_to_plot, condition=condition, feature='distance')
        pref_angles = self.extract_condition_feature(cells_to_plot, condition=condition, feature='RF')

        theta = np.radians(pref_angles)
        pref_distances = np.array(pref_distances) + np.random.randn(len(pref_distances)) * 0.1 + 1
        sc = ax.scatter(theta, pref_distances, c=pref_angles, cmap='viridis', vmin=-30, vmax=60, s=40, edgecolor='k', alpha=0.8)

        ax.grid(False)
        ax.set_theta_direction(-1)
        ax.set_thetamin(-35)
        ax.set_thetamax(65)
        ax.set_ylabel('Preferred Distance')
        ax.set_yticks([])
        # title will be condition
        if condition is None:
            ax.set_title('Responsive', fontsize=18, color='black', pad=2)
        else:
            ax.set_title(condition.capitalize(), fontsize=18, color='black', pad=2)

        ax.legend()

        plt.tight_layout()

        return fig, ax

    def plot_preferred_distance_polar_density(self):
        """
        Plot a density plot of preferred distances (4 bins) vs. preferred angles (binned),
        as a radial pcolormesh for each condition.
        Rows: distances (4 bins), Columns: angles (binned).
        """
        import matplotlib as mpl

        # Define bins
        distance_bins = np.array([1, 2, 3, 4, 5])  # 4 bins: [1-2), [2-3), [3-4), [4-5)
        angle_min, angle_max = -30, 75
        n_angle_bins = 18
        angle_bins = np.linspace(angle_min, angle_max, n_angle_bins + 1)  # e.g. 6 deg per bin

        # For each condition, build a 2D histogram: (distance_bin, angle_bin)
        condition_colors = {
            'binocular': 'Greens',
            'monocular': 'Reds'
        }

        fig = plt.figure(figsize=(10, 5))
        n_conds = len(condition_colors)
        for i, (cond, cmap_name) in enumerate(condition_colors.items()):
            # Prepare data
            pref_distances = []
            pref_angles = []
            if cond in self.modulated_cells:
                mod_cells = self.modulated_cells[cond]
                # Find the corresponding stat_df for this condition
                stat_df = None
                for df in self.radial_stats:
                    if not df.empty and df['condition'].iloc[0] == cond:
                        stat_df = df
                        break
                if stat_df is not None:
                    for cell in mod_cells:
                        row = stat_df[stat_df['cell_id'] == cell]
                        if not row.empty:
                            # Add small noise to avoid binning artifacts
                            noisy_distance = row['pref_distance'].values[0] + np.random.uniform(-0.2, 0.2)
                            noisy_angle = row['rf_position'].values[0] + np.random.uniform(-1, 1)
                            pref_distances.append(noisy_distance + 1)  # +1 to match original code
                            pref_angles.append(noisy_angle)
            # Bin data
            H, r_edges, theta_edges = np.histogram2d(
                pref_distances, pref_angles,
                bins=[distance_bins, angle_bins]
            )
            # Prepare meshgrid for pcolormesh
            # r: [distance_bins], theta: [angle_bins] in radians
            r, theta = np.meshgrid(distance_bins, np.radians(angle_bins), indexing='ij')
            # H shape: (4, n_angle_bins)
            # For pcolormesh, need (r.shape[0], r.shape[1])
            # Plot
            ax = fig.add_subplot(1, n_conds, i+1, polar=True)
            # pcolormesh expects theta (angle) as 2D array, r as 2D array, and H.T as values
            # But meshgrid above is (4, n_angle_bins+1), H is (4, n_angle_bins)
            # So we need to transpose H for correct orientation
            pcm = ax.pcolormesh(
                np.radians(angle_bins),  # theta edges
                distance_bins,           # r edges
                H,                       # shape (4, n_angle_bins)
                cmap=cmap_name,
                shading='auto'
            )
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_thetamin(angle_min)
            ax.set_thetamax(angle_max)
            ax.set_ylim(distance_bins[0], distance_bins[-1])
            ax.set_yticks(distance_bins[:-1] + 0.5)
            ax.set_yticklabels([f"{int(distance_bins[j])}-{int(distance_bins[j+1])}" for j in range(len(distance_bins)-1)])
            ax.set_xticks(np.radians(np.linspace(angle_min, angle_max, 8)))
            ax.set_title(f"{cond.capitalize()}", fontsize=14)
            cbar = fig.colorbar(pcm, ax=ax, pad=0.1, orientation='vertical', shrink=0.8)
            cbar.set_label('Cell Density', fontsize=10)
            if i == 0:
                ax.set_ylabel('Preferred Distance Bin')
        fig.suptitle('Density of Preferred Distance vs. Angle', fontsize=18, color='black', y=0.98)
        plt.tight_layout()
        return fig

    def plot_reliable_cells_map(self, ax=None, show_areas=False):
        """
        Plot map of reliable cells colored by RF location.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to show visual area boundaries (multi-session only).

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        # Get all cell IDs to plot
        cells_to_plot = self.session_df['cell_id'].values

        # Get background image and display
        bg_img = self._get_background_image()
        vmin = np.percentile(bg_img, 1)
        vmax = np.percentile(bg_img, 99)
        extent = self._get_background_extent()

        ax.imshow(bg_img, alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax,
                  extent=extent, origin='upper')

        # Get coordinates for all cells
        x_coords, y_coords = self._get_cell_coordinates(cells_to_plot)

        # Scatter plot each cell
        for i, cell_id in enumerate(cells_to_plot):
            xpix, ypix = x_coords[i], y_coords[i]

            if cell_id not in self.reliable_classes['reliable']:
                ax.scatter(
                    xpix, ypix,
                    c='lightgray',
                    s=120, alpha=0.5,
                    edgecolor='black', linewidth=1
                )
            else:
                rf_location = self.extract_general_feature(cell=[cell_id], feature='RF')
                ax.scatter(
                    xpix, ypix,
                    c=rf_location,
                    cmap='viridis',
                    vmin=-30,
                    vmax=60,
                    s=120, alpha=1,
                    edgecolor='black', linewidth=1
                )

        # Show area boundaries
        if show_areas:
            self._overlay_area_boundaries(ax)

        self._apply_axis_adjustments(ax)

        # Remove all axes, spines, ticks
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax

    def plot_rf_density_map(self, bins=20, ax=None, show_areas=False):
        """
        Plot a 2D map where each spatial bin is colored by the average RF location
        of the cells whose centers lie in that bin.

        Parameters
        ----------
        bins : int
            Number of bins along the x and y dimensions.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to show visual area boundaries (multi-session only).

        Returns
        -------
        fig
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()

        # Get background image
        ref_img = self._get_background_image()

        # Gather (x, y) centroids and RF values for reliable cells
        reliable_cells = self.reliable_classes['reliable']
        xs, ys = self._get_cell_coordinates(reliable_cells)
        rfvals = np.array([self.extract_general_feature(cell=[cell_id], feature='RF')[0]
                          for cell_id in reliable_cells])

        # Compute mean RF in each 2D bin
        mean_rf, x_edges, y_edges, binnumber = binned_statistic_2d(
            xs, ys, rfvals,
            statistic='mean',
            bins=[bins, bins],
            expand_binnumbers=False
        )
        # Compute count per bin
        count_rf, _, _, _ = binned_statistic_2d(
            xs, ys, rfvals,
            statistic='count',
            bins=[bins, bins],
            expand_binnumbers=False
        )
        # Mask bins with <=1 sample
        mean_rf[count_rf <= 1] = np.nan

        # Show background reference image
        vmin = np.percentile(ref_img, 1)
        vmax = np.percentile(ref_img, 99)
        extent = self._get_background_extent()

        ax.imshow(
            ref_img,
            cmap='gray',
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
            origin='lower' if self._invert_axes else 'upper',
            extent=[0, ref_img.shape[1], 0, ref_img.shape[0]] if self._invert_axes else extent
        )

        # Overlay the binned RF map
        im = ax.imshow(
            mean_rf.T,
            cmap='viridis',
            alpha=0.8,
            vmin=-30, vmax=75,
            interpolation='nearest',
            origin='upper',
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]]
        )
        cbar = fig.colorbar(im, ax=ax, label='Mean RF Location', shrink=0.8)

        # Show area boundaries
        if show_areas:
            self._overlay_area_boundaries(ax)

        ax.set_title(f'{self._get_session_title()} — RF Density Map', fontsize=16, color='black')
        ax.axis('off')
        plt.tight_layout(pad=0)

        return fig

    def plot_modulated_cells_map(self, condition='binocular', ax=None, show_areas=False):
        """
        Plot map of modulated cells colored by preferred distance.

        Parameters
        ----------
        condition : str
            Condition to plot ('binocular', 'monocular', 'both', 'none')
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to show visual area boundaries (multi-session only).

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        # Get background image and display
        bg_img = self._get_background_image()
        vmin = np.percentile(bg_img, 1)
        vmax = np.percentile(bg_img, 99)
        extent = self._get_background_extent()

        ax.imshow(bg_img, alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax,
                  extent=extent, origin='upper')

        # Get cells for this condition
        modulated_cells = self.modulated_classes[condition]
        none_cells = self.modulated_classes['none']

        # Get coordinates for none cells and plot in grey
        if len(none_cells) > 0:
            none_x, none_y = self._get_cell_coordinates(none_cells)
            ax.scatter(none_x, none_y, c='lightgray', s=120, alpha=0.5,
                       edgecolor='black', linewidth=1)

        # Get coordinates for modulated cells and plot colored by distance
        if len(modulated_cells) > 0:
            mod_x, mod_y = self._get_cell_coordinates(modulated_cells)
            if condition is not None:
                distance_values = self.extract_condition_feature(modulated_cells, condition=condition, feature='distance')
            else:
                distance_values = self.extract_general_feature(modulated_cells, feature='distance')
            ax.scatter(mod_x, mod_y, c=distance_values, cmap='viridis', vmin=0, vmax=3,
                       s=120, alpha=1, edgecolor='black', linewidth=1)

        # Show area boundaries
        if show_areas:
            self._overlay_area_boundaries(ax)

        self._apply_axis_adjustments(ax)

        # Remove axes and spines
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'{condition.capitalize() if condition else "All Reliable"} Cells', fontsize=16)

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax

    def plot_cell_features(self, condition='binocular', feature='distance',
                           ax=None, show_areas=True, cmap='viridis'):
        """
        Plot cells on the background map color-coded by a feature.

        Parameters
        ----------
        condition : str
            Condition to filter cells ('monocular', 'binocular', 'both', 'reliable').
            Use 'reliable' to show all reliable cells without modulation filtering.
        feature : str
            Feature to color by ('distance', 'DTI', etc.). For condition='reliable',
            use general features like 'RF'.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to show visual area boundaries.
        cmap : str, optional
            Colormap for feature values (default 'viridis')

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        # Get background image and display
        bg_img = self._get_background_image()
        bg_vmin = np.percentile(bg_img, 1)
        bg_vmax = np.percentile(bg_img, 99)
        extent = self._get_background_extent()

        ax.imshow(bg_img, alpha=0.5, cmap='gray', vmin=bg_vmin, vmax=bg_vmax,
                  extent=extent, origin='upper')

        # Get cells based on condition
        reliable_df = self.filter_by_reliability(self.session_df, reliable_type='reliable', threshold=0.85)

        if condition == 'reliable':
            # Use all reliable cells without modulation filtering
            cells_df = reliable_df
            # Plot unreliable cells in grey as background
            bg_df = self.filter_by_reliability(self.session_df, reliable_type='not_reliable', threshold=0.85)
            if len(bg_df) > 0:
                bg_x, bg_y = self._get_cell_coordinates(bg_df)
                ax.scatter(bg_x, bg_y, c='lightgray', s=85, alpha=0.5, edgecolor='black', linewidth=1)
        else:
            # Filter by modulation type
            cells_df = self.filter_by_modulation(reliable_df, condition)
            # Plot reliable but not modulated cells in grey as background
            bg_df = reliable_df.loc[reliable_df.index.difference(cells_df.index)]
            if len(bg_df) > 0:
                bg_x, bg_y = self._get_cell_coordinates(bg_df)
                ax.scatter(bg_x, bg_y, c='lightgray', s=85, alpha=0.5, edgecolor='black', linewidth=1)

        n_colored = len(cells_df)
        n_gray = len(bg_df)
        n_total = n_colored + n_gray
        pct_colored = (n_colored / n_total * 100) if n_total > 0 else 0
        print(f'[{condition}] colored: {n_colored}, gray: {n_gray}, total: {n_total} ({pct_colored:.1f}% colored)')

        if len(cells_df) == 0:
            ax.set_title(f'No {condition} cells found', fontsize=16)
            return fig, ax

        # Get coordinates using existing method
        mod_x, mod_y = self._get_cell_coordinates(cells_df)

        # Get feature values based on condition
        if condition == 'reliable':
            # Use general feature extraction for reliable cells
            feature_values = self.extract_general_feature(
                input_df=cells_df,
                feature=feature
            )
        else:
            # Use condition-specific feature extraction
            feature_values = self.extract_condition_feature(
                input_df=cells_df,
                condition=condition,
                feature=feature
            )

        # Set vmin/vmax based on feature
        if feature == 'distance':
            vmin, vmax = 0, 3
        elif feature == 'RF':
            vmin, vmax = -30, 60
        else:
            vmin, vmax = np.nanmin(feature_values), np.nanmax(feature_values)

        # Plot all cells at once
        scatter = ax.scatter(
            mod_x, mod_y,
            c=feature_values,
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            s=85, alpha=1,
            edgecolor='black', linewidth=1
        )

        # # Add colorbar
        # cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        # cbar.set_label(feature.capitalize(), fontsize=14)

        # Show area boundaries
        if show_areas:
            self._overlay_area_boundaries(ax)

        self._apply_axis_adjustments(ax)

        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title
        if condition == 'reliable':
            condition_label = 'Reliable'
        elif condition == 'both':
            condition_label = 'Both'
        else:
            condition_label = condition.capitalize()

        # fig.suptitle(f'{condition_label} Cells - {feature.capitalize()}', fontsize=22, color='black', y=0.96, fontweight='bold')
        # ax.set_title(self._get_session_title(), fontsize=20, color='grey', pad=2)

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax

    def plot_cell_density(self, condition='reliable', feature='RF',
                          n_bins=20, min_count=1,
                          ax=None, show_areas=True, cmap='viridis'):
        """
        Plot 2D density of a feature across cortex by binning cells and
        coloring each bin by the mean feature value.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        bg_img = self._get_background_image()
        bg_vmin = np.percentile(bg_img, 1)
        bg_vmax = np.percentile(bg_img, 99)
        extent = self._get_background_extent()

        ax.imshow(bg_img, alpha=0.5, cmap='gray', vmin=bg_vmin, vmax=bg_vmax,
                  extent=extent, origin='upper')

        reliable_df = self.filter_by_reliability(self.session_df, reliable_type='reliable', threshold=0.85)
        if condition == 'reliable':
            cells_df = reliable_df
        else:
            cells_df = self.filter_by_modulation(reliable_df, condition)

        if len(cells_df) == 0:
            ax.set_title(f'No {condition} cells found', fontsize=16)
            return fig, ax

        x_min, x_max = sorted([extent[0], extent[1]])
        y_min, y_max = sorted([extent[2], extent[3]])
        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

        if feature == 'pctg':
            cond_x, cond_y = self._get_cell_coordinates(cells_df)
            if condition == 'reliable':
                denom_df = self.session_df
            else:
                denom_df = reliable_df
            denom_x, denom_y = self._get_cell_coordinates(denom_df)

            cond_x = np.asarray(cond_x, dtype=float)
            cond_y = np.asarray(cond_y, dtype=float)
            denom_x = np.asarray(denom_x, dtype=float)
            denom_y = np.asarray(denom_y, dtype=float)

            cond_count, _, _ = np.histogram2d(cond_x, cond_y, bins=[x_edges, y_edges])
            total_count, _, _ = np.histogram2d(denom_x, denom_y, bins=[x_edges, y_edges])

            with np.errstate(divide='ignore', invalid='ignore'):
                mean_grid = np.where(total_count > 0, cond_count / total_count * 100, np.nan)
            count_grid = total_count
        else:
            x, y = self._get_cell_coordinates(cells_df)
            if condition == 'reliable':
                feature_values = self.extract_general_feature(input_df=cells_df, feature=feature)
            else:
                feature_values = self.extract_condition_feature(input_df=cells_df, condition=condition, feature=feature)

            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            feature_values = np.asarray(feature_values, dtype=float)

            mean_grid, _, _, _ = binned_statistic_2d(
                x, y, feature_values, statistic=np.nanmean, bins=[x_edges, y_edges]
            )
            count_grid, _, _, _ = binned_statistic_2d(
                x, y, feature_values, statistic='count', bins=[x_edges, y_edges]
            )

        mean_grid = np.ma.masked_where((count_grid < min_count) | np.isnan(mean_grid), mean_grid)

        if feature == 'distance':
            vmin, vmax = 0.5, 2.5
            cmap = 'viridis_r'
        elif feature == 'RF':
            vmin, vmax = -30, 60
        elif feature == 'pctg':
            vmin = 0
            vmax = 50 if condition in ('binocular', 'monocular') else 100
            cmap = {'reliable': 'Greens', 'binocular': 'Blues', 'monocular': 'Reds'}.get(condition, cmap)
        else:
            vmin, vmax = np.nanmin(mean_grid), np.nanmax(mean_grid)

        mesh = ax.pcolormesh(
            x_edges, y_edges, mean_grid.T,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading='flat', alpha=0.85, edgecolors='none',
        )

        cbar_label = f'% {condition}' if feature == 'pctg' else feature.capitalize()
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(cbar_label, fontsize=14)

        if show_areas:
            self._overlay_area_boundaries(ax)

        self._apply_axis_adjustments(ax)

        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax

    def plot_location_map(self, condition, ax=None, show_areas=False):
        """
        Plot cells colored by modulation status.

        Parameters
        ----------
        condition : str
            Condition to check modulation for ('binocular', 'monocular')
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to show visual area boundaries (multi-session only).

        Returns
        -------
        fig
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        # Get background image and display
        bg_img = self._get_background_image()
        vmin = np.percentile(bg_img, 1)
        vmax = np.percentile(bg_img, 99)
        extent = self._get_background_extent()

        ax.imshow(bg_img, alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax,
                  extent=extent, origin='upper')

        # Get modulated cells for this condition
        modulated_cells = set(self.modulated_classes[condition])
        reliable_cells = self.reliable_classes['reliable']

        # Get coordinates for reliable cells
        xs, ys = self._get_cell_coordinates(reliable_cells)

        # Scatter plot each cell
        for i, cell_id in enumerate(reliable_cells):
            is_modulated = cell_id in modulated_cells

            if is_modulated:
                ax.scatter(xs[i], ys[i], c='yellow', s=100, alpha=1)
            else:
                ax.scatter(xs[i], ys[i], c='blue', s=100, alpha=1)

        # Show area boundaries
        if show_areas:
            self._overlay_area_boundaries(ax)

        self._apply_axis_adjustments(ax)

        ax.axis('off')
        plt.tight_layout(pad=0)

        return fig

    def plot_session_summary(self):

        has_monocular = self.session_df['monocular'].notna().any()
        has_binocular = self.session_df['binocular'].notna().any()
        available_conditions = []
        if has_monocular:
            available_conditions.append('monocular')
        if has_binocular:
            available_conditions.append('binocular')

        figs = {}

        figs['map_all'], _ = self.plot_modulated_cells_map(condition=None)
        for cond in available_conditions:
            figs[f'map_{cond}'], _ = self.plot_modulated_cells_map(condition=cond)

        n_conds = len(available_conditions)
        fig_distance_polar = plt.figure(figsize=(6 * n_conds, 6))
        gs = fig_distance_polar.add_gridspec(1, n_conds, wspace=0.5)
        for i, cond in enumerate(available_conditions):
            ax = fig_distance_polar.add_subplot(gs[0, i], projection='polar')
            self.plot_preferred_distance_polar(condition=cond, ax=ax)
        figs['distance_polar'] = fig_distance_polar

        for cond in available_conditions:
            figs[f'distance_bars_{cond}'] = self.plot_distance_bars(condition=cond)

        return figs

    def print_summary(self):
        """
        Print a text summary of cell statistics for each session.

        For each session prints: total cells, reliable cells, modulated cells per
        category (only for available conditions), and preferred distance distribution.
        """
        sessions = self.get_available_sessions()

        for session_name in sessions:
            session_df = self.filter_by_session(self.session_df, session_name=session_name)

            has_monocular = session_df['monocular'].notna().any()
            has_binocular = session_df['binocular'].notna().any()

            if has_monocular and has_binocular:
                mod_types = ['monocular', 'binocular', 'both', 'none']
            elif has_binocular:
                mod_types = ['binocular', 'none']
            else:
                mod_types = ['monocular', 'none']

            available_conditions = []
            if has_monocular:
                available_conditions.append('monocular')
            if has_binocular:
                available_conditions.append('binocular')

            total = len(session_df)
            reliable_df = self.filter_by_reliability(session_df)
            n_reliable = len(reliable_df)

            print(f"{'='*50}")
            print(f"Session: {session_name}")
            print(f"{'='*50}")
            print(f"Total cells:    {total}")
            print(f"Reliable cells: {n_reliable} / {total} ({100*n_reliable/total:.1f}%)")
            print()

            print("Modulated cells (% of reliable):")
            for mod_type in mod_types:
                mod_df = self.filter_by_modulation(session_df, modulation_type=mod_type)
                n_mod = len(mod_df)
                pct = 100 * n_mod / n_reliable if n_reliable > 0 else 0
                print(f"  {mod_type:<12}: {n_mod:>4} / {n_reliable} ({pct:.1f}%)")
            print()

            print("Preferred distance (% of modulated cells):")
            for cond in available_conditions:
                mod_df = self.filter_by_modulation(session_df, modulation_type=cond)
                n_mod = len(mod_df)
                if n_mod == 0:
                    print(f"  [{cond}] no modulated cells")
                    continue
                dist_values = self.extract_condition_feature(mod_df, condition=cond, feature='distance')
                dist_rounded = [int(round(d)) for d in dist_values]
                print(f"  [{cond}]")
                for d in range(4):
                    n_d = dist_rounded.count(d)
                    pct = 100 * n_d / n_mod
                    print(f"    Distance {d}: {n_d:>4} / {n_mod} ({pct:.1f}%)")
            print()


class SingleSessionVisualizer(SessionVisualizerBase):
    """
    Visualizer for single-session DataFrames using Suite2P coordinates.

    Plots cells on the Suite2P mean image background. Axes are inverted to
    match the Suite2P coordinate system.

    Parameters
    ----------
    session_df : pd.DataFrame
        Session DataFrame with cell statistics. Must contain exactly one unique
        'session_name' value and a 'cell_id' column.
    base_data_dir : str, optional
        Base directory for processed session data (Suite2P).

    Examples
    --------
    >>> viz = SingleSessionVisualizer(session_df)
    >>> viz.plot_reliable_cells_map()
    >>> viz.plot_modulated_cells_map(condition='binocular')
    """

    def __init__(self, session_df,
                 base_data_dir=r'D:\data-distance-coding-project\processed-data'):

        if 'session_name' not in session_df.columns:
            raise ValueError("DataFrame must contain 'session_name' column.")
        if session_df['session_name'].nunique() != 1:
            raise ValueError(
                "SingleSessionVisualizer requires a DataFrame with exactly one unique session_name. "
                f"Found {session_df['session_name'].nunique()} unique sessions. "
                "Use MultiSessionVisualizer for multi-session data."
            )

        super().__init__(session_df, base_data_dir)

        self.session_name = session_df['session_name'].iloc[0]
        self._suite2p_data = self._load_suite2p_data(self.session_name)

    def _load_suite2p_data(self, session_name):
        """
        Load Suite2P data for the session.

        Parameters
        ----------
        session_name : str
            Session identifier (e.g., '251128_SAM041_tiling_3_120um')

        Returns
        -------
        dict
            Dictionary with 'ops' and 'stat' arrays
        """
        suite2p_dir = os.path.join(self.base_data_dir, session_name, 'suite2p', 'plane0')

        ops_path = os.path.join(suite2p_dir, 'ops.npy')
        stat_path = os.path.join(suite2p_dir, 'stat.npy')

        if not os.path.exists(ops_path):
            raise FileNotFoundError(f"Suite2P ops.npy not found at {ops_path}")
        if not os.path.exists(stat_path):
            raise FileNotFoundError(f"Suite2P stat.npy not found at {stat_path}")

        ops = np.load(ops_path, allow_pickle=True).item()
        stat = np.load(stat_path, allow_pickle=True)

        return {'ops': ops, 'stat': stat}

    def _get_background_image(self):
        return self._suite2p_data['ops']['refImg']

    def _get_background_extent(self):
        img = self._get_background_image()
        return [0, img.shape[1], img.shape[0], 0]

    def _get_cell_coordinates(self, input_df):
        """
        Get (x, y) coordinates for cells from Suite2P stat array.

        Parameters
        ----------
        input_df : pd.DataFrame
            DataFrame with 'cell_id' column.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (x_coords, y_coords) arrays
        """
        cell_ids = input_df['cell_id'].values
        stat = self._suite2p_data['stat']
        x_coords = []
        y_coords = []
        for cell_id in cell_ids:
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            x_coords.append(xpix)
            y_coords.append(ypix)
        return np.array(x_coords), np.array(y_coords)

    def _get_session_title(self):
        return self.session_name

    def _apply_axis_adjustments(self, ax):
        ax.invert_xaxis()
        ax.invert_yaxis()

    @property
    def _invert_axes(self):
        return True

    def plot_mean_traces(self, condition='both', ax=None, device=None):
        """
        Plot mean traces for modulated cells.

        Note: Requires raw trace data via a RadialDevice instance.

        Parameters
        ----------
        condition : str
            Condition to plot (unused, plots all conditions)
        ax : matplotlib.axes.Axes, optional
            Unused parameter kept for API compatibility
        device : RadialDevice, optional
            Device to get traces from. Required for this method.

        Raises
        ------
        ValueError
            If device is not provided.
        """
        if device is None:
            raise ValueError(
                "plot_mean_traces requires a 'device' (RadialDevice) parameter "
                "to access raw trace data."
            )

        # Create subplots for modulation classes excluding 'none'
        conditions_to_plot = [cond for cond in self.modulated_classes.keys() if cond != 'none']
        fig, axes = plt.subplots(1, len(conditions_to_plot), figsize=(5*len(conditions_to_plot), 6))

        # Handle case where there's only one condition (axes won't be an array)
        if len(conditions_to_plot) == 1:
            axes = [axes]

        # Find the maximum number of cells across all conditions (excluding 'none') to normalize y-axis
        max_cells = max([len(self.modulated_classes[condition]) for condition in conditions_to_plot])

        for i, condition in enumerate(conditions_to_plot):
            modulated_cells = self.modulated_classes[condition]

            if len(modulated_cells) == 0:
                axes[i].text(0.5, 0.5, 'No cells', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(condition.capitalize(), fontsize=14, color=self.modulation_colors[condition], fontweight='bold')
                continue

            RF = self.extract_condition_feature(cell=modulated_cells, condition=condition, feature='RF')

            all_mean_traces = []
            for cell in modulated_cells:
                mean_trace = device.get_all_traces(cell=cell, mean_trace=True)[0]
                all_mean_traces.append(mean_trace)

            all_mean_traces = np.array(all_mean_traces)

            # order RF from lowest to highest and apply same order to all_mean_traces
            sorted_indices = np.argsort(RF)
            RF = RF[sorted_indices]
            all_mean_traces = all_mean_traces[sorted_indices]

            # plot all_mean_traces - create meshgrid for pcolormesh
            X, Y = np.meshgrid(device.angle_range, np.arange(len(all_mean_traces)))
            axes[i].pcolormesh(X, Y, all_mean_traces, shading='auto')
            axes[i].set_title(condition.capitalize(), fontsize=14, color=self.modulation_colors[condition], fontweight='bold')
            axes[i].set_xlabel('Angle (degrees)', fontsize=12)
            axes[i].set_ylabel('Cell (sorted by RF)', fontsize=12)

            # Set y-axis limits to match the maximum number of cells
            axes[i].set_ylim(0, max_cells)

            # Remove axis spines and ticks
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)

            plt.tight_layout()

        # Create separate figure for 'none' condition
        if 'none' in self.modulated_classes.keys():
            fig_none, ax_none = plt.subplots(1, 1, figsize=(5, 6))

            modulated_cells = self.modulated_classes['none']

            if len(modulated_cells) == 0:
                ax_none.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax_none.transAxes)
                ax_none.set_title('None', fontsize=14, color=self.modulation_colors['none'], fontweight='bold')
            else:
                RF = self.extract_condition_feature(cell=modulated_cells, condition='none', feature='RF')

                all_mean_traces = []
                for cell in modulated_cells:
                    mean_trace = device.get_all_traces(cell=cell, mean_trace=True)[0]
                    all_mean_traces.append(mean_trace)

                all_mean_traces = np.array(all_mean_traces)

                # order RF from lowest to highest and apply same order to all_mean_traces
                sorted_indices = np.argsort(RF)
                RF = RF[sorted_indices]
                all_mean_traces = all_mean_traces[sorted_indices]

                # plot all_mean_traces - create meshgrid for pcolormesh
                X, Y = np.meshgrid(device.angle_range, np.arange(len(all_mean_traces)))
                ax_none.pcolormesh(X, Y, all_mean_traces, shading='auto')
                ax_none.set_title('None', fontsize=14, color=self.modulation_colors['none'], fontweight='bold')
                ax_none.set_xlabel('Angle (degrees)', fontsize=12)
                ax_none.set_ylabel('Cell (sorted by RF)', fontsize=12)

            plt.tight_layout()

    def plot_rf_histogram(self, condition='binocular', subset='modulated', bins=20, ax=None, color='steelblue'):
        
        """
        Parameters
        ----------
        condition : str   'binocular', 'monocular', 'both', 'none'
        subset    : str   'modulated' (cells modulated for condition) or 'reliable' (all reliable cells)
        """
        if subset == 'modulated':
            df = self.modulated_classes[condition]
            rf_vals = self.extract_condition_feature(df, condition=condition, feature='RF')
        else:
            df = self.reliable_classes['reliable']
            rf_vals = self.extract_condition_feature(df, condition=condition, feature='RF')

        rf_vals = rf_vals[~np.isnan(rf_vals)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        ax.hist(rf_vals, bins=bins, color=color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('RF Position (°)', fontsize=16)
        ax.set_ylabel('Number of Cells', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)
        fig.suptitle(f'RF Distribution — {condition.capitalize()} ({subset})', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig


    def save_session_cells(self, device, results_dir):
        """
        Plot and save all reliable cells grouped by modulation type.

        Creates one subfolder per modulation category (binocular, monocular, both, none)
        under results_dir and saves one figure per cell using
        CellVisualizer.plot_all_traces_colormap_distances.

        Parameters
        ----------
        device : RadialDevice
            Device instance used by CellVisualizer to access raw traces.
        results_dir : str
            Base directory where category subfolders will be created.
        """
        cell_viz = CellVisualizer(device)
        categories = ['binocular', 'monocular', 'both', 'none']

        for category in categories:
            category_df = self.modulated_classes[category]
            if len(category_df) == 0:
                print(f'{category}: no cells, skipping.')
                continue

            category_dir = create_new_result_dir(results_dir, category)
            cell_ids = category_df['cell_id'].values
            print(f'{category}: saving {len(cell_ids)} cells...')

            for cell_id in cell_ids:
                fig = cell_viz.plot_all_traces_colormap_distances(cell_id)
                fig.savefig(os.path.join(category_dir, f'cell_{cell_id}.png'),
                            bbox_inches='tight', dpi=150)
                plt.close(fig)


class MultiSessionVisualizer(SessionVisualizerBase):
    """
    Visualizer for multi-session DataFrames using widefield coordinates.

    Plots cells on the widefield image background. Includes area boundary overlays
    and methods for comparing across sessions and visual areas.

    Parameters
    ----------
    session_df : pd.DataFrame
        Session DataFrame with cell statistics. Must contain 'session_name' and
        'wf_coords' columns, and multiple unique session names.
    mouse_name : str
        Mouse identifier (e.g., 'SAM041'). Required.
    experiment_name : str
        Experiment name (e.g., 'tiling_120um'). Required.
    base_data_dir : str, optional
        Base directory for processed session data.
    base_multi_dir : str, optional
        Base directory for multi-session/widefield data.

    Examples
    --------
    >>> viz = MultiSessionVisualizer(combined_df, mouse_name='SAM041', experiment_name='tiling_120um')
    >>> viz.plot_reliable_cells_map(show_areas=True)
    >>> viz.plot_modulation_fraction_by_angle(modulation_type='binocular', area='V1')
    """

    def __init__(self, session_df, mouse_name, experiment_name,
                 base_data_dir=r'D:\data-distance-coding-project\processed-data',
                 base_multi_dir=r'D:\data-distance-coding-project\multi-session-data'):

        if 'session_name' not in session_df.columns:
            raise ValueError("DataFrame must contain 'session_name' column.")
        if mouse_name is None or experiment_name is None:
            raise ValueError(
                "MultiSessionVisualizer requires 'mouse_name' and 'experiment_name' parameters."
            )
        if 'wf_coords' not in session_df.columns:
            raise ValueError(
                "MultiSessionVisualizer requires 'wf_coords' column in session_df. "
                "Use SessionCombiner with widefield_mapper to add widefield coordinates."
            )

        super().__init__(session_df, base_data_dir)

        self.mouse_name = mouse_name
        self.experiment_name = experiment_name
        self.base_multi_dir = base_multi_dir
        self._widefield_data = self._load_widefield_data()

    def _load_widefield_data(self):
        """
        Load widefield alignment data.

        Returns
        -------
        dict
            Dictionary with 'wf_maps', 'areas_verts', and 'wf_transformations'
        """
        multi_session_dir = os.path.join(
            self.base_multi_dir,
            f'{self.mouse_name}_{self.experiment_name}'
        )
        params_path = os.path.join(
            multi_session_dir,
            f'{self.mouse_name}_{self.experiment_name}_alignment_parameters.pkl'
        )

        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Widefield alignment parameters not found at {params_path}. "
                "Run WidefieldAligner first to create alignment."
            )

        with open(params_path, 'rb') as f:
            aligner_params = pickle.load(f)

        return aligner_params

    def _get_background_image(self):
        return self._widefield_data['wf_maps']['img']

    def _get_background_extent(self):
        return [0, 400, 400, 0]

    def _get_cell_coordinates(self, input_df):
        """
        Get (x, y) coordinates for cells from the 'wf_coords' column.

        Parameters
        ----------
        input_df : pd.DataFrame
            DataFrame with 'wf_coords' column (each entry is [x, y]).

        Returns
        -------
        tuple of (array-like, array-like)
            (x_coords, y_coords)
        """
        wf_coords = input_df['wf_coords'].values
        x_coords, y_coords = zip(*wf_coords)
        return x_coords, y_coords

    def _get_session_title(self):
        session_names = self.session_df['session_name'].unique()
        mouse_names = list(set([s.split('_')[1] for s in session_names]))
        n_sessions = len(session_names)
        if len(mouse_names) == 1:
            return f"{mouse_names[0]} ({n_sessions} sessions)"
        else:
            return f"{', '.join(mouse_names)} ({n_sessions} sessions)"

    def _overlay_area_boundaries(self, ax, show_legend=True, color='black'):
        """Overlay visual area boundaries on a plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        show_legend : bool, optional
            Whether to show legend for area boundaries (default True).
        color : str, optional
            Edge color for area boundaries (default 'black').
        """
        areas_verts = self._widefield_data['areas_verts']

        # Define preferred order for areas
        area_order = ['V1', 'LM', 'RL', 'AM', 'PM', 'AL', 'None']

        # Sort areas by preferred order, putting unknown areas at the end
        sorted_areas = sorted(
            areas_verts.items(),
            key=lambda x: area_order.index(x[0]) if x[0] in area_order else len(area_order)
        )

        for i, (area_name, area_verts) in enumerate(sorted_areas):
            patch = Polygon(area_verts, closed=True, facecolor='none',
                           edgecolor=color, linewidth=3, label=area_name, alpha=0.8)
            ax.add_patch(patch)

        # if show_legend:
        #     ax.legend(loc='upper right', fontsize=14, framealpha=0.9, edgecolor='gray')

    def _get_area_colors(self, group_by):
        if group_by != 'area':
            return {}
        return {'V1': 'tab:blue', 'LM': 'tab:green', 'RL': 'tomato',
                'AM': 'tab:orange', 'PM': 'tab:purple', 'AL': 'tab:brown',
                'None': 'tab:gray'}

    def get_available_areas(self):
        """
        Get list of all available visual area names in preferred order.

        Areas are returned in order: V1, LM, RL, AM, PM, AL, None, then any others.

        Returns
        -------
        list
            List of area name strings.
        """
        available_areas = self.session_df['area'].unique().tolist()
        if np.nan in available_areas:
            available_areas.remove(np.nan)

        # Define preferred order for areas
        area_order = ['V1', 'LM', 'RL', 'AM', 'PM', 'AL', 'None']

        # Sort areas by preferred order, putting unknown areas at the end
        sorted_areas = sorted(
            available_areas,
            key=lambda x: area_order.index(x) if x in area_order else len(area_order)
        )

        return sorted_areas

    def _get_azi_to_rf_regression(self, df):
        """
        Fit a linear regression from widefield azimuth to binocular RF.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to use (already filtered by area/session if needed).

        Returns
        -------
        slope : float
        intercept : float
        """
        reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
        azi_vals = reliable_df['azi'].values
        rf_vals = self.extract_condition_feature(reliable_df, condition='binocular', feature='RF')
        rf_vals = np.array(rf_vals)

        mask = ~np.isnan(rf_vals) & ~np.isnan(azi_vals)
        slope, intercept = np.polyfit(azi_vals[mask], rf_vals[mask], 1)
        return slope, intercept

    def plot_reliability_scatter(self, condition='binocular', rf_axis='azi',
                                 area=None, session_name=None, ax=None):
        """
        Scatter plot of retinotopic position vs reliability.

        Parameters
        ----------
        condition : str
            Condition to plot ('monocular' or 'binocular').
        rf_axis : str
            Retinotopic axis for X ('azi' or 'alt').
        area : str, optional
            Filter by visual area (e.g. 'V1', 'LM', 'RL').
        session_name : str, optional
            Filter by session name.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        # Filter cells
        df = self.session_df
        if area is not None:
            df = df[df['area'] == area]
        if session_name is not None:
            df = df[df['session_name'] == session_name]

        # Rescale azi to 0-130 range
        if rf_axis == 'azi':
            all_vals = df[rf_axis].values
            azi_min, azi_max = all_vals.min(), all_vals.max()

        # Split into reliable and unreliable
        reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
        unreliable_df = df.loc[df.index.difference(reliable_df.index)]

        # Extract data and scatter unreliable first (grey, behind)
        if len(unreliable_df) > 0:
            x_unrel = unreliable_df[rf_axis].values
            if rf_axis == 'azi':
                x_unrel = (x_unrel - azi_min) / (azi_max - azi_min) * 130
            y_unrel = unreliable_df[condition].apply(lambda x: x['reliable_corr']).values
            ax.scatter(x_unrel, y_unrel, s=20, alpha=0.4, color='lightgrey',
                       edgecolor='none', label='Unreliable')

        # Then reliable on top (colored)
        if len(reliable_df) > 0:
            x_rel = reliable_df[rf_axis].values
            if rf_axis == 'azi':
                x_rel = (x_rel - azi_min) / (azi_max - azi_min) * 130
            y_rel = reliable_df[condition].apply(lambda x: x['reliable_corr']).values
            ax.scatter(x_rel, y_rel, s=20, alpha=0.6, edgecolor='none', label='Reliable')

        # Reliability threshold line
        ax.axhline(y=0.85, color='red', linestyle='--', linewidth=1, label='Threshold (0.85)')

        # Labels
        axis_label = 'Azimuth' if rf_axis == 'azi' else 'Altitude'
        ax.set_xlabel(f'{axis_label} (deg)', fontsize=14)
        ax.set_ylabel('Reliability (corr)', fontsize=14)
        ax.legend(fontsize=10)

        filter_parts = [condition.capitalize()]
        if area is not None:
            filter_parts.append(area)
        if session_name is not None:
            filter_parts.append(session_name)
        filter_str = ' | '.join(filter_parts)
        fig.suptitle(f'{filter_str} — Reliability vs {axis_label}', fontsize=18, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)

        plt.tight_layout()

        return fig, ax

    def plot_rf_vs_azi(self, condition='binocular', area=None, session_name=None, ax=None):
        """
        Scatter plot comparing RF (condition-specific) to azimuth for reliable cells.

        Parameters
        ----------
        condition : str
            Condition for RF extraction ('monocular' or 'binocular').
        area : str, optional
            Filter by visual area.
        session_name : str, optional
            Filter by session name.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.get_figure()

        # Filter cells
        df = self.session_df
        if area is not None:
            df = df[df['area'] == area]
        if session_name is not None:
            df = df[df['session_name'] == session_name]

        # Get reliable cells only
        reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)

        if len(reliable_df) == 0:
            ax.set_title('No reliable cells found', fontsize=16)
            return fig, ax

        # X: azi values
        azi_vals = reliable_df['azi'].values

        # Y: condition-specific RF
        rf_vals = self.extract_condition_feature(reliable_df, condition=condition, feature='RF')

        # Scatter
        ax.scatter(azi_vals, rf_vals, s=20, alpha=0.6, edgecolor='none')

        # Regression line with correlation stats
        mask = ~np.isnan(rf_vals)
        if mask.sum() > 1:
            slope, intercept = np.polyfit(azi_vals[mask], rf_vals[mask], 1)
            r, pval = pearsonr(azi_vals[mask], rf_vals[mask])
            r2 = r ** 2
            x_line = np.array([azi_vals[mask].min(), azi_vals[mask].max()])
            ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5,
                    label=f'y = {slope:.2f}x + {intercept:.2f}\nr = {r:.3f}, R² = {r2:.3f}, p = {pval:.2e}')
            ax.legend(fontsize=10)

        # Labels
        ax.set_xlabel('Widefield Azimuth', fontsize=14)
        ax.set_ylabel(f'Device RF — {condition.capitalize()}', fontsize=14)

        filter_parts = [condition.capitalize()]
        if area is not None:
            filter_parts.append(area)
        if session_name is not None:
            filter_parts.append(session_name)
        filter_str = ' | '.join(filter_parts)
        fig.suptitle(f'{filter_str} — Device RF vs Widefield Azimuth', fontsize=18, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)

        plt.tight_layout()

        return fig, ax

    def plot_reliability_fraction(self, rf_axis='alt', axis_range=None,
                                  window_size=5, area=None, session_name=None,
                                  ax=None, show_density=True, color='steelblue'):
        """
        Plot fraction of reliable cells as a function of retinotopic position using moving bins.

        Parameters
        ----------
        rf_axis : str
            Retinotopic axis to bin over ('alt' or 'azi').
        axis_range : tuple, optional
            (min, max) range. If None, derived from data.
        window_size : float, optional
            Half-width of the sliding window (default: 5).
        area : str, optional
            Filter by visual area.
        session_name : str, optional
            Filter by session name.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.
        show_density : bool, optional
            Whether to overlay total cell count on secondary y-axis.
        color : str, optional
            Color for the fraction line.

        Returns
        -------
        fig, results
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        # Filter by area/session
        df = self.session_df
        if area is not None:
            df = self.filter_by_area(df, area_name=area)
        if session_name is not None:
            df = df[df['session_name'] == session_name]

        # Get all cells and reliable cells positions
        all_vals = df[rf_axis].values.copy()
        reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
        reliable_vals = reliable_df[rf_axis].values.copy()

        # Map azi values to RF space using regression
        if rf_axis == 'azi':
            slope, intercept = self._get_azi_to_rf_regression(df)
            all_vals = slope * all_vals + intercept
            reliable_vals = slope * reliable_vals + intercept

        # Determine range
        if axis_range is None:
            axis_range = (int(np.floor(all_vals.min())), int(np.ceil(all_vals.max())))

        # Sliding window
        positions = np.arange(axis_range[0], axis_range[1] + 1, 1)
        fractions = []
        counts_total = []
        counts_reliable = []

        for pos in positions:
            total = np.sum((all_vals >= pos - window_size) & (all_vals < pos + window_size))
            reliable = np.sum((reliable_vals >= pos - window_size) & (reliable_vals < pos + window_size))
            counts_total.append(total)
            counts_reliable.append(reliable)
            fractions.append(reliable / total if total > 0 else np.nan)

        # Plot fraction
        ax.plot(positions, fractions, color=color, lw=2, label='Reliable fraction')
        if rf_axis == 'azi':
            axis_label = 'RF (from Azimuth regression)'
        else:
            axis_label = 'Altitude'
        ax.set_xlabel(f'{axis_label}', fontsize=14)
        ax.set_ylabel('Fraction Reliable', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
        ax.grid(alpha=0.3)

        # Density overlay
        if show_density:
            ax2 = ax.twinx()
            ax2.fill_between(positions, counts_total, color='gray', alpha=0.2, label='Total count')
            ax2.set_ylabel('Total Cell Count', fontsize=14, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

        # Title
        filter_parts = []
        if area is not None:
            filter_parts.append(area)
        if session_name is not None:
            filter_parts.append(session_name)
        filter_str = f' - {" | ".join(filter_parts)}' if filter_parts else ''
        fig.suptitle(f'Reliable Fraction by {axis_label}{filter_str} (window: ±{window_size})',
                     fontsize=18, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)

        plt.tight_layout()

        results = {
            'positions': positions,
            'fractions': np.array(fractions),
            'counts_total': np.array(counts_total),
            'counts_reliable': np.array(counts_reliable)
        }

        return fig, results

    def plot_modulation_fraction_by_angle(self, modulation_type='binocular',
                                          angle_range=(-30, 60), window_size=5,
                                          area=None, ax=None, show_density=True, color=None):
        """
        Plot fraction of modulated cells among reliable cells as a function of RF angle.

        Parameters
        ----------
        modulation_type : str
            Type of modulation ('binocular', 'monocular', 'both')
        angle_range : tuple, optional
            (min_angle, max_angle) range to plot (default: (-30, 60))
        window_size : float, optional
            Half-width of the sliding window in degrees (default: 5)
        area : str, optional
            Visual area to filter by (e.g., 'V1', 'LM'). If None, uses all areas.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_density : bool, optional
            Whether to overlay reliable cell count on secondary y-axis (default: True)
        color : str, optional
            Color for the line plot. If None, uses modulation_colors dict.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        results : dict
            Dictionary with 'angles', 'fractions', 'counts_reliable', 'counts_modulated'
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        if color is None:
            color = self.modulation_colors.get(modulation_type, 'steelblue')

        # Filter by area if specified
        if area is not None:
            base_df = self.filter_by_area(self.session_df, area_name=area)
        else:
            base_df = self.session_df

        # Get reliable cells and their RF values
        reliable_df = self.filter_by_reliability(base_df, reliable_type='reliable', threshold=0.85)
        reliable_rf = self.extract_general_feature(reliable_df, feature='RF')

        # Get modulated cells and their RF values
        modulated_df = self.filter_by_modulation(base_df, modulation_type=modulation_type)
        # Extract RF values - use condition-specific extraction if modulation type is specified
        if modulation_type in ['binocular', 'monocular']:
            modulated_rf = self.extract_condition_feature(modulated_df, condition=modulation_type, feature='RF')
        else:
            modulated_rf = self.extract_general_feature(modulated_df, feature='RF')

        # Convert to numpy arrays
        reliable_rf = np.array(reliable_rf)
        modulated_rf = np.array(modulated_rf)

        # Create angle bins
        angles = np.arange(angle_range[0], angle_range[1] + 1, 1)

        # Compute fractions for each angle window
        frac_modulated = []
        counts_reliable = []
        counts_modulated = []

        for ang in angles:
            # Count reliable cells in window
            rf_window = (reliable_rf >= ang - window_size) & (reliable_rf < ang + window_size)
            rel_count = np.sum(rf_window)

            # Count modulated cells in window
            mod_count = np.sum((modulated_rf >= ang - window_size) & (modulated_rf < ang + window_size))

            counts_reliable.append(rel_count)
            counts_modulated.append(mod_count)

            if rel_count > 0:
                frac = mod_count / rel_count
            else:
                frac = np.nan
            frac_modulated.append(frac)

        # Plot fraction
        ax.plot(angles, frac_modulated, color=color, lw=5, label=f'{modulation_type.capitalize()} fraction')
        ax.set_xlabel('RF Angle (degrees)', fontsize=14)
        ax.set_ylabel(f'Fraction {modulation_type.capitalize()} among Reliable', fontsize=14)
        ax.set_ylim(0, 1.05)
        # ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
        ax.grid(alpha=0.3)
        #set lims from 0 to 0.5
        ax.set_ylim(0, 0.6)

        # Overlay reliable cell density
        if show_density:
            ax2 = ax.twinx()
            ax2.fill_between(angles, counts_reliable, color='gray', alpha=0.2, label='Reliable count')
            ax2.set_ylabel('Reliable Cell Count', fontsize=14, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

        # Title
        area_label = f' - {area}' if area is not None else ''
        fig.suptitle(f'{modulation_type.capitalize()} Fraction by RF Angle{area_label} (window: ±{window_size}°)',
                     fontsize=18, color='black', y=0.96, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)

        plt.tight_layout()

        results = {
            'angles': angles,
            'fractions': np.array(frac_modulated),
            'counts_reliable': np.array(counts_reliable),
            'counts_modulated': np.array(counts_modulated)
        }


        return fig, results

    def plot_cell_features_composite(self, condition='binocular', feature='distance',
                                     session_names=None, cmap='viridis',
                                     ax=None, show_areas=True):
        """
        Plot cells color-coded by a feature on the widefield image, masked to only
        show pixels covered by session footprints. Area boundaries overlaid in white.

        Parameters
        ----------
        condition : str
            Condition to filter cells ('monocular', 'binocular', 'both', 'reliable').
        feature : str
            Feature to color by ('distance', 'DTI', 'RF', etc.).
        session_names : list of str, optional
            Sessions to include. If None, uses all sessions in wf_transformations.
        cmap : str, optional
            Colormap for feature values (default 'viridis').
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        show_areas : bool, optional
            Whether to overlay area boundaries in white.

        Returns
        -------
        fig, ax
        """
        import cv2

        def _transform_image(img, scale, rotation, dx, dy):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotation, scale)
            M[0, 2] += dx
            M[1, 2] += dy
            return cv2.warpAffine(img, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        wf_transformations = self._widefield_data['wf_transformations']
        if session_names is None:
            session_names = list(wf_transformations.keys())

        # Build union mask from all session footprints
        combined_mask = np.zeros((400, 400), dtype=bool)

        for session_name in session_names:
            if session_name not in wf_transformations:
                print(f"Warning: {session_name} not found in wf_transformations. Skipping.")
                continue
            params = wf_transformations[session_name]
            ops_path = os.path.join(self.base_data_dir, session_name, 'suite2p', 'plane0', 'ops.npy')
            try:
                ops = np.load(ops_path, allow_pickle=True).item()
                ref_img = ops['refImg']
            except FileNotFoundError:
                print(f"Warning: Could not load refImg for {session_name}. Skipping.")
                continue

            ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
            ref_resized = cv2.resize(ref_norm, (400, 400), interpolation=cv2.INTER_LINEAR)
            transformed = _transform_image(ref_resized,
                                           params['scale'], params['rotation'],
                                           params['dx'], params['dy'])
            combined_mask |= (transformed > 0)

        # Apply mask to widefield image
        wf_img = self._widefield_data['wf_maps']['img'].astype(float)
        wf_norm = (wf_img - wf_img.min()) / (wf_img.max() - wf_img.min())
        outside = (~combined_mask).astype(float)
        masked_wf = wf_norm * combined_mask + outside
        bg_rgb = np.stack([masked_wf, masked_wf, masked_wf], axis=2)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        ax.imshow(bg_rgb, extent=[0, 400, 400, 0], origin='upper', alpha=0.5)

        # Filter cells (same logic as plot_cell_features)
        reliable_df = self.filter_by_reliability(self.session_df, reliable_type='reliable', threshold=0.85)

        if condition == 'reliable':
            cells_df = reliable_df
            unreliable_df = self.filter_by_reliability(self.session_df, reliable_type='not_reliable', threshold=0.85)
            if len(unreliable_df) > 0:
                bg_x, bg_y = self._get_cell_coordinates(unreliable_df)
                ax.scatter(bg_x, bg_y, c='lightgray', s=50, alpha=1, edgecolor='black', linewidth=1)
        else:
            cells_df = self.filter_by_modulation(reliable_df, condition)
            non_mod_df = reliable_df.loc[reliable_df.index.difference(cells_df.index)]
            if len(non_mod_df) > 0:
                bg_x, bg_y = self._get_cell_coordinates(non_mod_df)
                ax.scatter(bg_x, bg_y, c='lightgray', s=50, alpha=1, edgecolor='black', linewidth=1)

        if len(cells_df) == 0:
            ax.set_title(f'No {condition} cells found', fontsize=16)
            return fig, ax

        mod_x, mod_y = self._get_cell_coordinates(cells_df)

        if condition == 'reliable':
            feature_values = self.extract_general_feature(input_df=cells_df, feature=feature)
        else:
            feature_values = self.extract_condition_feature(input_df=cells_df, condition=condition, feature=feature)

        if feature == 'distance':
            cell_colors = [self.distance_colors[int(v)] for v in feature_values]
            ax.scatter(mod_x, mod_y, c=cell_colors, s=50, alpha=1,
                       edgecolor='black', linewidth=1)
        else:
            if feature == 'RF':
                vmin, vmax = -30, 60
            else:
                vmin, vmax = np.nanmin(feature_values), np.nanmax(feature_values)
            ax.scatter(mod_x, mod_y, c=feature_values, cmap=cmap,
                       vmin=vmin, vmax=vmax, s=50, alpha=1,
                       edgecolor='black', linewidth=1)

        if show_areas:
            self._overlay_area_boundaries(ax, color='black')

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.tight_layout(pad=0)

        return fig

    def plot_distance_pref_vs_rf(self, condition='both', area=None,
                                 x_axis='RF', bin_width=10.0,
                                 min_cells_per_bin=20,
                                 axis_range=None, ax=None):
        """
        Distribution of preferred distance across bins of an x-axis variable.

        For each bin of `x_axis`, plots the percentage of cells preferring each
        of the 4 distances (0-3). Within each bin the 4 lines sum to 100%. Bins
        with fewer than `min_cells_per_bin` cells are dropped (NaN, so the line
        breaks rather than connecting across them).

        Parameters
        ----------
        condition : str
            'monocular', 'binocular', 'both', 'reliable', or 'un-modulated'.
        area : str, optional
            Filter cells by visual area (e.g. 'V1').
        x_axis : str
            'RF' (degrees, condition-aware), 'azi', or 'alt'. For 'azi'/'alt'
            the raw column is linearly rescaled to the canonical range using
            the full `session_df` min/max (so the scale doesn't change with
            condition or area filtering).
        bin_width : float
            Bin width in degrees (default 10.0).
        min_cells_per_bin : int
            Drop bins with fewer total cells than this (default 20).
        axis_range : tuple of (float, float), optional
            (min, max) range for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        df = self.session_df
        if area is not None:
            filt_df = self.filter_by_area(df, area_name=area)
        else:
            filt_df = df

        if condition == 'reliable':
            target_df = self.filter_by_reliability(filt_df, reliable_type='reliable', threshold=0.85)
            dist_vals = self.extract_general_feature(target_df, feature='distance')
        elif condition == 'un-modulated':
            target_df = self.filter_by_modulation(filt_df, modulation_type='none')
            dist_vals = self.extract_general_feature(target_df, feature='distance')
        else:
            target_df = self.filter_by_modulation(filt_df, modulation_type=condition)
            dist_vals = self.extract_condition_feature(target_df, condition=condition, feature='distance')

        x_vals = self._get_x_axis_values(target_df, x_axis, condition)

        x_vals = np.asarray(x_vals, dtype=float)
        dist_vals = np.asarray(dist_vals, dtype=float)
        valid = ~(np.isnan(x_vals) | np.isnan(dist_vals))
        x_vals = x_vals[valid]
        dist_vals = np.round(dist_vals[valid]).astype(int)

        distances = [0, 1, 2, 3]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        if len(x_vals) == 0:
            ax.set_title('No cells found', fontsize=16)
            return fig, ax

        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)

        counts = (pd.crosstab(bin_idx, dist_vals)
                  .reindex(index=range(len(centers)), columns=distances, fill_value=0))

        totals = counts.sum(axis=1)
        if (totals >= min_cells_per_bin).sum() == 0:
            ax.set_title(f'No bins with >= {min_cells_per_bin} cells', fontsize=16)
            return fig, ax

        pct = counts.div(totals.replace(0, np.nan), axis=0) * 100
        pct.loc[totals < min_cells_per_bin, :] = np.nan

        for j, dist in enumerate(distances):
            ax.plot(centers, pct[dist].values, marker='o', linewidth=2,
                    color=self.distance_colors[j], label=f'Distance {dist}')

        for k, total in enumerate(totals.values):
            if total >= min_cells_per_bin:
                top = np.nanmax(pct.iloc[k].values)
                ax.text(centers[k], top + 2, f'n={int(total)}',
                        ha='center', va='bottom', fontsize=10, color='black')

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        ax.set_ylabel('Percentage of Cells (%)', fontsize=16)
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        condition_label = {'reliable': 'Reliable', 'un-modulated': 'Un-modulated',
                           'both': 'Combined'}.get(condition, condition.capitalize())
        title_parts = [condition_label]
        if area is not None:
            title_parts.append(area)
        fig.suptitle(f'Preferred Distance vs {x_axis} ({" | ".join(title_parts)})',
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

    def plot_distance_pref_vs_rf_compare(self, groups,
                                         x_axis='RF', bin_width=10.0,
                                         min_cells_per_bin=20,
                                         axis_range=None, rf_range=None,
                                         pct_of_distance=None,
                                         show_stats=True, ax=None):
        """
        Mean continuous preferred distance vs `x_axis`, one line per
        (area, condition) group, pooled across all sessions of this animal.
        Error bars are SEM across cells in each bin (`std(ddof=1) / sqrt(n)`).

        Use for cross-group comparison (e.g. V1 vs RL; V1-binocular vs
        V1-monocular). No per-session split — all cells in a group are pooled.

        Parameters
        ----------
        groups : list of (str or None, str)
            (area, condition) tuples. `area=None` means no area filter.
            `condition` is 'monocular', 'binocular', 'both', 'reliable', or
            'un-modulated'.
        x_axis : str
            'RF', 'azi', or 'alt'.
        bin_width : float
            Bin width in degrees.
        min_cells_per_bin : int
            Bins below this cell count become NaN (line breaks).
        axis_range : tuple of (float, float), optional
            (min, max) for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        rf_range : tuple of (float, float), optional
            Condition-aware RF window filter.
        pct_of_distance : int or None
            If None (default), y is mean continuous distance per bin (0–3
            axis). If 0/1/2/3, y becomes the % of cells in each bin whose
            (rounded) preferred distance equals this value (0–100 axis);
            error bars are binomial SEM `sqrt(p*(1-p)/n) * 100`. Per-bin
            stats switch from Mann-Whitney U to Fisher's exact.
        show_stats : bool
            If True and >= 2 groups have data, runs a per-bin test on the
            first two groups (Mann-Whitney U in mean mode; Fisher's exact
            in pct mode) — only on bins with >= `min_cells_per_bin` cells
            in both — corrects p-values across bins with Benjamini-Hochberg
            (alpha=0.05), and marks surviving bins with a star above the
            point. Descriptive only — does not constitute a single global
            claim about the curves.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        n_bins = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        area_color_map = self._get_area_colors('area')
        fallback_cycle = list(plt.cm.tab10.colors)
        category_colors = {'binocular': 'tab:blue', 'monocular': 'tab:red',
                           'un-modulated': 'gray', 'none': 'gray'}
        color_by_category = len({a for (a, _) in groups}) == 1
        any_plotted = False
        group_data = []

        for gi, (area, condition) in enumerate(groups):
            df = self.session_df
            if area is not None:
                df = self.filter_by_area(df, area_name=area)

            if condition == 'reliable':
                target_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
            elif condition == 'un-modulated':
                target_df = self.filter_by_modulation(df, modulation_type='none')
            else:
                target_df = self.filter_by_modulation(df, modulation_type=condition)

            target_df = self._filter_by_rf_range(target_df, condition, rf_range)

            if condition in ('reliable', 'un-modulated'):
                dist_vals = self.extract_general_feature(target_df, feature='distance')
            else:
                dist_vals = self.extract_condition_feature(target_df, condition=condition, feature='distance')

            x_vals = self._get_x_axis_values(target_df, x_axis, condition)

            x_vals = np.asarray(x_vals, dtype=float)
            dist_vals = np.asarray(dist_vals, dtype=float)
            valid = ~(np.isnan(x_vals) | np.isnan(dist_vals))
            x_vals = x_vals[valid]
            dist_vals = dist_vals[valid]

            if len(x_vals) == 0:
                group_data.append(None)
                continue

            bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
            bin_idx_arr = np.asarray(bin_idx, dtype=float)

            if pct_of_distance is None:
                df_tmp = pd.DataFrame({'bin': bin_idx, 'dist': dist_vals})
                grouped = df_tmp.groupby('bin')['dist']
                means = grouped.mean().reindex(range(n_bins)).values
                stds = grouped.std(ddof=1).reindex(range(n_bins)).values
                n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
                with np.errstate(invalid='ignore', divide='ignore'):
                    sems = stds / np.sqrt(n_per)
            else:
                is_d = (np.round(dist_vals).astype(int) == pct_of_distance).astype(float)
                df_tmp = pd.DataFrame({'bin': bin_idx, 'is_d': is_d})
                grouped = df_tmp.groupby('bin')['is_d']
                p_hat = grouped.mean().reindex(range(n_bins)).values
                n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
                means = p_hat * 100
                with np.errstate(invalid='ignore', divide='ignore'):
                    sems = np.sqrt(p_hat * (1 - p_hat) / n_per) * 100

            mask = n_per < min_cells_per_bin
            means[mask] = np.nan
            sems[mask] = np.nan

            if color_by_category:
                color = category_colors.get(condition, fallback_cycle[gi % len(fallback_cycle)])
            else:
                color = area_color_map.get(area) if area is not None else None
                if color is None:
                    color = fallback_cycle[gi % len(fallback_cycle)]

            if color_by_category and pct_of_distance is not None and condition in ('monocular', 'binocular', 'both'):
                color = self.distance_colors[pct_of_distance]

            label = f'{area if area is not None else "any"} {condition}'

            ax.errorbar(centers, means, yerr=sems, marker='o', linewidth=2,
                        capsize=3, color=color, label=label)
            any_plotted = True

            bin_dist = {k: dist_vals[bin_idx_arr == k] for k in range(n_bins)}
            group_data.append({'means': means, 'sems': sems,
                               'n_per': n_per, 'bin_dist': bin_dist})

        if not any_plotted:
            ax.set_title('No cells for any group', fontsize=16)
            return fig, ax

        if show_stats:
            valid_groups = [g for g in group_data if g is not None]
            if len(valid_groups) >= 2:
                g1, g2 = valid_groups[0], valid_groups[1]
                pvals = np.full(n_bins, np.nan)
                for k in range(n_bins):
                    if (g1['n_per'][k] >= min_cells_per_bin and
                            g2['n_per'][k] >= min_cells_per_bin):
                        a = g1['bin_dist'][k]
                        b = g2['bin_dist'][k]
                        try:
                            if pct_of_distance is None:
                                _, p = mannwhitneyu(a, b, alternative='two-sided')
                            else:
                                a_d = int(np.sum(np.round(a).astype(int) == pct_of_distance))
                                b_d = int(np.sum(np.round(b).astype(int) == pct_of_distance))
                                table = [[a_d, len(a) - a_d],
                                         [b_d, len(b) - b_d]]
                                _, p = fisher_exact(table, alternative='two-sided')
                            pvals[k] = p
                        except ValueError:
                            pass

                tested = ~np.isnan(pvals)
                if tested.any():
                    p_sub = pvals[tested]
                    n_tests = len(p_sub)
                    sorted_p = np.sort(p_sub)
                    thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
                    under = sorted_p <= thresholds
                    cutoff = sorted_p[np.where(under)[0].max()] if under.any() else -np.inf
                    sig_sub = p_sub <= cutoff

                    sig = np.zeros(n_bins, dtype=bool)
                    sig[tested] = sig_sub

                    for k in range(n_bins):
                        if sig[k]:
                            tops = []
                            for g in (g1, g2):
                                m = g['means'][k]
                                s = g['sems'][k] if not np.isnan(g['sems'][k]) else 0.0
                                if not np.isnan(m):
                                    tops.append(m + s)
                            if tops:
                                star_offset = 0.1 if pct_of_distance is None else 5.0
                                ax.text(centers[k], max(tops) + star_offset, '*',
                                        ha='center', va='bottom',
                                        fontsize=18, color='black')

        y_max_curves = 0.0
        for g in group_data:
            if g is None:
                continue
            sems_safe = np.where(np.isnan(g['sems']), 0.0, g['sems'])
            vals = g['means'] + sems_safe
            if not np.all(np.isnan(vals)):
                y_max_curves = max(y_max_curves, float(np.nanmax(vals)))

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        if pct_of_distance is None:
            ax.set_ylabel('Mean Distance', fontsize=16)
            ax.set_ylim(0, min(3, y_max_curves + 0.2))
            suptitle = f'Mean Distance vs {x_axis} (comparison)'
        else:
            ax.set_ylabel(f'% at Distance {pct_of_distance}', fontsize=16)
            ax.set_ylim(0, min(100, y_max_curves + 10.0))
            suptitle = f'% at Distance {pct_of_distance} vs {x_axis} (comparison)'
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        fig.suptitle(suptitle,
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

    def plot_modulation_by_rf_compare(self, groups,
                                      x_axis='RF', bin_width=10.0,
                                      min_cells_per_bin=20,
                                      axis_range=None, rf_range=None,
                                      show_stats=True, ax=None):
        """
        % of reliable cells of a given modulation category vs `x_axis`,
        one line per (area, modulation_type) group, pooled across all
        sessions of this animal. Error bars are binomial SEM
        `sqrt(p*(1-p)/n) * 100`.

        Denominator per bin is all reliable cells in that (area, RF) bin;
        numerator is the subset matching `modulation_type`.

        Parameters
        ----------
        groups : list of (str or None, str)
            (area, modulation_type) tuples. `area=None` means no area
            filter. `modulation_type` is 'monocular', 'binocular', 'both',
            or 'un-modulated' (treated as 'none').
        x_axis : str
            'RF', 'azi', or 'alt'.
        bin_width : float
            Bin width in degrees.
        min_cells_per_bin : int
            Bins below this reliable-cell count become NaN (line breaks).
        axis_range : tuple of (float, float), optional
            (min, max) for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        rf_range : tuple of (float, float), optional
            RF window filter applied to the reliable pool (general RF).
        show_stats : bool
            If True and >= 2 groups have data, runs a per-bin Fisher's
            exact test on the first two groups — only on bins with >=
            `min_cells_per_bin` reliable cells in both — corrects across
            bins with Benjamini-Hochberg (alpha=0.05), and marks
            surviving bins with a star above the point. Descriptive only.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        n_bins = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        area_color_map = self._get_area_colors('area')
        fallback_cycle = list(plt.cm.tab10.colors)
        category_colors = {'binocular': 'tab:blue', 'monocular': 'tab:red',
                           'un-modulated': 'gray', 'none': 'gray'}
        color_by_category = len({a for (a, _) in groups}) == 1
        any_plotted = False
        group_data = []

        for gi, (area, modulation_type) in enumerate(groups):
            df = self.session_df
            if area is not None:
                df = self.filter_by_area(df, area_name=area)

            reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
            reliable_df = self._filter_by_rf_range(reliable_df, 'reliable', rf_range)

            mod_key = 'none' if modulation_type == 'un-modulated' else modulation_type
            mod_df = self.filter_by_modulation(reliable_df, modulation_type=mod_key)

            x_vals = self._get_x_axis_values(reliable_df, x_axis, 'reliable')
            x_vals = np.asarray(x_vals, dtype=float)
            is_mod = reliable_df.index.isin(mod_df.index).astype(float)

            valid = ~np.isnan(x_vals)
            x_vals = x_vals[valid]
            is_mod = is_mod[valid]

            if len(x_vals) == 0:
                group_data.append(None)
                continue

            bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
            bin_idx_arr = np.asarray(bin_idx, dtype=float)

            df_tmp = pd.DataFrame({'bin': bin_idx, 'is_mod': is_mod})
            grouped = df_tmp.groupby('bin')['is_mod']
            p_hat = grouped.mean().reindex(range(n_bins)).values
            n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
            means = p_hat * 100
            with np.errstate(invalid='ignore', divide='ignore'):
                sems = np.sqrt(p_hat * (1 - p_hat) / n_per) * 100

            mask = n_per < min_cells_per_bin
            means[mask] = np.nan
            sems[mask] = np.nan

            if color_by_category:
                color = category_colors.get(modulation_type, fallback_cycle[gi % len(fallback_cycle)])
            else:
                color = area_color_map.get(area) if area is not None else None
                if color is None:
                    color = fallback_cycle[gi % len(fallback_cycle)]

            label = f'{area if area is not None else "any"} {modulation_type}'

            ax.errorbar(centers, means, yerr=sems, marker='o', linewidth=2,
                        capsize=3, color=color, label=label)
            any_plotted = True

            bin_mod = {k: is_mod[bin_idx_arr == k] for k in range(n_bins)}
            group_data.append({'means': means, 'sems': sems,
                               'n_per': n_per, 'bin_mod': bin_mod})

        if not any_plotted:
            ax.set_title('No cells for any group', fontsize=16)
            return fig, ax

        if show_stats:
            valid_groups = [g for g in group_data if g is not None]
            if len(valid_groups) >= 2:
                g1, g2 = valid_groups[0], valid_groups[1]
                pvals = np.full(n_bins, np.nan)
                for k in range(n_bins):
                    if (g1['n_per'][k] >= min_cells_per_bin and
                            g2['n_per'][k] >= min_cells_per_bin):
                        a = g1['bin_mod'][k]
                        b = g2['bin_mod'][k]
                        try:
                            a_m = int(np.sum(a == 1))
                            b_m = int(np.sum(b == 1))
                            table = [[a_m, len(a) - a_m],
                                     [b_m, len(b) - b_m]]
                            _, p = fisher_exact(table, alternative='two-sided')
                            pvals[k] = p
                        except ValueError:
                            pass

                tested = ~np.isnan(pvals)
                if tested.any():
                    p_sub = pvals[tested]
                    n_tests = len(p_sub)
                    sorted_p = np.sort(p_sub)
                    thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
                    under = sorted_p <= thresholds
                    cutoff = sorted_p[np.where(under)[0].max()] if under.any() else -np.inf
                    sig_sub = p_sub <= cutoff

                    sig = np.zeros(n_bins, dtype=bool)
                    sig[tested] = sig_sub

                    for k in range(n_bins):
                        if sig[k]:
                            tops = []
                            for g in (g1, g2):
                                m = g['means'][k]
                                s = g['sems'][k] if not np.isnan(g['sems'][k]) else 0.0
                                if not np.isnan(m):
                                    tops.append(m + s)
                            if tops:
                                ax.text(centers[k], max(tops) + 5.0, '*',
                                        ha='center', va='bottom',
                                        fontsize=18, color='black')

        y_max_curves = 0.0
        for g in group_data:
            if g is None:
                continue
            sems_safe = np.where(np.isnan(g['sems']), 0.0, g['sems'])
            vals = g['means'] + sems_safe
            if not np.all(np.isnan(vals)):
                y_max_curves = max(y_max_curves, float(np.nanmax(vals)))

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        ax.set_ylabel('% of Reliable Cells', fontsize=16)
        ax.set_ylim(0, min(100, y_max_curves + 10.0))
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        fig.suptitle(f'% Modulated Cells vs {x_axis} (comparison)',
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax


class MultiAnimalVisualizer(SessionVisualizerBase):
    """
    Visualizer for combined multi-animal DataFrames.

    Takes a DataFrame built by combine_animals() and provides population-level
    plots with bootstrap statistics across animals.

    Parameters
    ----------
    all_df : pd.DataFrame
        Combined DataFrame from combine_animals(). Must include an 'animal' column
        and the standard session DataFrame columns (RF, monocular, binocular, etc.).
    """

    def __init__(self, all_df, base_data_dir=r'D:\data-distance-coding-project\processed-data'):
        super().__init__(all_df, base_data_dir=base_data_dir)

    def _get_background_image(self):
        return None

    def _get_background_extent(self):
        return None

    def _get_cell_coordinates(self, input_df):
        return None, None

    def _get_session_title(self):
        if 'animal' in self.session_df.columns:
            n_animals = self.session_df['animal'].nunique()
            return f'{n_animals} animals'
        return ''

    def get_available_areas(self):
        available_areas = self.session_df['area'].dropna().unique().tolist()
        area_order = ['V1', 'LM', 'RL', 'AM', 'PM', 'AL', 'None']
        return sorted(available_areas, key=lambda x: area_order.index(x) if x in area_order else len(area_order))

    def _get_area_colors(self, group_by):
        if group_by != 'area':
            return {}
        return {'V1': 'tab:blue', 'LM': 'tab:green', 'RL': 'tomato',
                'AM': 'tab:orange', 'PM': 'tab:purple', 'AL': 'tab:brown',
                'None': 'tab:gray'}

    def plot_cell_features(self, experiment_type, condition='binocular', feature='distance',
                           show_areas=True, cmap='viridis'):
        """
        Plot cell features on the widefield map for each animal as a subplot.

        Parameters
        ----------
        experiment_type : str
            Experiment name used to load each animal's widefield data (e.g., 'tiling_120um').
        condition : str
            'monocular', 'binocular', 'both', or 'reliable'.
        feature : str
            Feature to color by ('distance', 'DTI', 'RF', etc.).
        show_areas : bool
            Whether to overlay visual area boundaries. Default True.
        cmap : str
            Colormap for feature values. Default 'viridis'.

        Returns
        -------
        matplotlib.figure.Figure
        """
        animals = sorted(self.session_df['animal'].unique())
        n = len(animals)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), dpi=100)
        axes = np.array(axes).flatten()

        for ax, animal in zip(axes, animals):
            animal_df = self.session_df[self.session_df['animal'] == animal].copy()
            try:
                viz = MultiSessionVisualizer(animal_df, mouse_name=animal, experiment_name=experiment_type)
                viz.plot_cell_features(condition=condition, feature=feature, ax=ax,
                                       show_areas=show_areas, cmap=cmap)
                ax.set_title(animal, fontsize=13)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, wrap=True)
                ax.set_title(animal, fontsize=13)
                ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        condition_label = 'Reliable' if condition == 'reliable' else condition.capitalize()
        fig.suptitle(f'{condition_label} Cells — {feature.capitalize()}',
                     fontsize=20, fontweight='bold', y=1.01)
        plt.tight_layout()
        return fig

    def plot_cell_density(self, experiment_type, condition='reliable', feature='RF',
                          n_bins=20, min_count=1,
                          show_areas=True, cmap='viridis'):
        """
        Plot 2D density of a feature on the widefield map for each animal as a subplot.
        """
        animals = sorted(self.session_df['animal'].unique())
        n = len(animals)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), dpi=100)
        axes = np.array(axes).flatten()

        for ax, animal in zip(axes, animals):
            animal_df = self.session_df[self.session_df['animal'] == animal].copy()
            try:
                viz = MultiSessionVisualizer(animal_df, mouse_name=animal, experiment_name=experiment_type)
                viz.plot_cell_density(condition=condition, feature=feature,
                                      n_bins=n_bins, min_count=min_count,
                                      ax=ax, show_areas=show_areas, cmap=cmap)
                ax.set_title(animal, fontsize=13)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, wrap=True)
                ax.set_title(animal, fontsize=13)
                ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        condition_label = 'Reliable' if condition == 'reliable' else condition.capitalize()
        feature_label = f'% {condition}' if feature == 'pctg' else feature.capitalize()
        fig.suptitle(f'{condition_label} Cells — {feature_label} density',
                     fontsize=20, fontweight='bold', y=1.01)
        plt.tight_layout()
        return fig

    def plot_widefield_with_areas_and_sessions(self, experiment_type, session_alpha=0.6):
        """
        Plot widefield map for each animal as a subplot in a single figure.

        Parameters
        ----------
        experiment_type : str
            Experiment name used to locate each animal's alignment parameters (e.g., 'tiling_120um').
        session_alpha : float
            Transparency for session overlays. Default 0.6.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from src.viz.WidefieldVisualizer import WidefieldVisualizer

        animals = sorted(self.session_df['animal'].unique())
        n = len(animals)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), dpi=100)
        axes = np.array(axes).flatten()

        for ax, animal in zip(axes, animals):
            try:
                wf = WidefieldVisualizer(animal, experiment_type)
                animal_sessions = self.session_df[self.session_df['animal'] == animal]['session_name'].unique().tolist()
                wf._draw_widefield_to_ax(ax, session_names=animal_sessions, session_alpha=session_alpha)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes, fontsize=9, wrap=True)
                ax.set_title(animal, fontsize=13)
                ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        return fig

    def plot_modulation_by_rf(self, n_bootstrap=1000, ax=None, viridis_colors=True):
        """
        Bar plot of modulation fraction grouped by modulation type, split by RF location.

        Three groups on x-axis (monocular, binocular, both). Within each group,
        two bars: one for cells with RF in the binocular region (-20° to 20°) and
        one for cells with RF in the monocular region (20° to 40°). Bootstrap CI
        error bars (95%) are shown on each bar.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations (default: 1000)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        viridis_colors : bool, optional
            If True (default), bar colors are sampled from viridis at the midpoint of each
            RF region, normalized over the full RF range (-30° to 60°).
            If False, uses the original colors: blue (#4C72B0) and orange (#DD8452).

        Returns
        -------
        matplotlib.figure.Figure
        """
        rf_regions = {
            'Binocular angles\n(-20° to 20°)': (-20, 20),
            'Monocular angles\n(20° to 40°)': (20, 40),
        }
        modulation_types = ['binocular', 'monocular', 'both']
        modulation_labels = ['Both eyes', 'Contralateral eye', 'Both conditions']

        if viridis_colors:
            rf_colors = [plt.cm.viridis(0.25), plt.cm.viridis(0.99)]  # dark blue, lighter green
        else:
            rf_colors = ['#4C72B0', '#DD8452']  # original: blue for binocular, orange for monocular

        # Pre-filter reliable cells per RF region and bootstrap
        region_stats = {}
        for region_name, (rf_min, rf_max) in rf_regions.items():
            rf_df = self.session_df[self.session_df['RF'].between(rf_min, rf_max)]
            reliable_df = self.filter_by_reliability(rf_df, reliable_type='reliable', threshold=0.85)
            n = len(reliable_df)

            boot_results = {m: [] for m in modulation_types}
            for _ in range(n_bootstrap):
                sample = reliable_df.sample(n=n, replace=True)
                for mod_type in modulation_types:
                    n_mod = len(self.filter_by_modulation(sample, modulation_type=mod_type))
                    boot_results[mod_type].append(n_mod / n * 100 if n > 0 else 0)

            region_stats[region_name] = {
                'n_reliable': n,
                'modulations': {
                    m: {
                        'mean': np.mean(boot_results[m]),
                        'ci_low': np.percentile(boot_results[m], 2.5),
                        'ci_high': np.percentile(boot_results[m], 97.5),
                    }
                    for m in modulation_types
                }
            }

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        regions = list(rf_regions.keys())
        n_regions = len(regions)
        bar_width = 0.30
        bar_gap = 0.04
        x = np.arange(len(modulation_types)) * 0.9

        for k, (region_name, color) in enumerate(zip(regions, rf_colors)):
            offset = (k - n_regions / 2 + 0.5) * (bar_width + bar_gap)
            for i, mod_type in enumerate(modulation_types):
                stats = region_stats[region_name]['modulations'][mod_type]
                mean = stats['mean']
                yerr_low = mean - stats['ci_low']
                yerr_high = stats['ci_high'] - mean

                ax.bar(x[i] + offset, mean,
                       width=bar_width,
                       color=color,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=region_name if i == 0 else '')
                ax.errorbar(x[i] + offset, mean,
                            yerr=[[yerr_low], [yerr_high]],
                            fmt='none', color='black', capsize=4, linewidth=1.5)

        # n= labels below each modulation group
        for i, mod_type in enumerate(modulation_types):
            ns = [region_stats[r]['n_reliable'] for r in regions]
            label = ' / '.join(f'n={n}' for n in ns)
            ax.text(x[i], -1.5, label, ha='center', va='top', fontsize=10, color='gray')

        ax.set_ylabel('% of Reliable Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(modulation_labels, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.axhline(5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)
        fig.suptitle('Modulation by RF Location', fontsize=22, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)
        plt.tight_layout()

        return fig

    def plot_distance_by_modulation(self, n_bootstrap=1000, ax=None):
        """
        Bar plot of preferred distance distribution for two cell populations, with bootstrap CI.

        Two groups on x-axis: reliable-but-not-modulated cells and binocular-modulated cells.
        Within each group, four bars show % of cells preferring each distance (0-3),
        with 95% bootstrap CI error bars.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations (default: 1000)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        distances = [0, 1, 2, 3]

        # Build cell groups
        reliable_df = self.filter_by_reliability(self.session_df, reliable_type='reliable', threshold=0.85)
        not_modulated_df = self.filter_by_modulation(reliable_df, modulation_type='none')
        binocular_df = self.filter_by_modulation(self.session_df, modulation_type='binocular')

        cell_groups = {
            'Reliable\n(not modulated)': (not_modulated_df, 'distance'),
            'Binocular\nmodulated': (binocular_df, 'binocular'),
        }

        def _get_pref_dists(df, source):
            if source == 'distance':
                return df['distance'].values.astype(float)
            else:
                return df[source].apply(lambda x: x['distance']).values.astype(float)

        # Bootstrap per group
        group_stats = {}
        for group_name, (group_df, dist_source) in cell_groups.items():
            n = len(group_df)
            boot_results = {d: [] for d in distances}
            for _ in range(n_bootstrap):
                sample = group_df.sample(n=n, replace=True)
                pref = [int(np.round(v)) for v in _get_pref_dists(sample, dist_source)]
                for dist in distances:
                    boot_results[dist].append(np.sum(np.array(pref) == dist) / n * 100 if n > 0 else 0)

            group_stats[group_name] = {
                'n': n,
                'distances': {
                    d: {
                        'mean': np.mean(boot_results[d]),
                        'ci_low': np.percentile(boot_results[d], 2.5),
                        'ci_high': np.percentile(boot_results[d], 97.5),
                    }
                    for d in distances
                }
            }

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        groups = list(group_stats.keys())
        n_dists = len(distances)
        bar_width = 0.20
        bar_gap = 0.03
        x = np.arange(len(groups)) * 1.3

        for j, dist in enumerate(distances):
            offset = (j - n_dists / 2 + 0.5) * (bar_width + bar_gap)
            for i, group_name in enumerate(groups):
                stats = group_stats[group_name]['distances'][dist]
                mean = stats['mean']
                yerr_low = mean - stats['ci_low']
                yerr_high = stats['ci_high'] - mean

                ax.bar(x[i] + offset, mean,
                       width=bar_width,
                       color=self.distance_colors[j],
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=f'Distance {dist}' if i == 0 else '')
                ax.errorbar(x[i] + offset, mean,
                            yerr=[[yerr_low], [yerr_high]],
                            fmt='none', color='black', capsize=3, linewidth=1.5)

        # n= labels
        for i, group_name in enumerate(groups):
            n_cells = group_stats[group_name]['n']
            ax.text(x[i], -1.5, f'n={n_cells}', ha='center', va='top', fontsize=11, color='gray')

        ax.set_ylabel('% of Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        fig.suptitle('Preferred Distance by Modulation Type', fontsize=22, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)
        plt.tight_layout()

        return fig

    def plot_modulation_per_area(self, n_bootstrap=1000, areas=None):
        """
        % of reliable cells modulated per brain area, monocular and binocular side by side,
        with 95% bootstrap CI.

        Pools all animals together per area and bootstraps from the whole pool
        (n = total reliable cells in that area across all animals).

        Parameters
        ----------
        n_bootstrap : int
            Bootstrap iterations. Default 1000.
        areas : list of str, optional
            Areas to include. Default ['V1', 'LM', 'RL'].
        """
        if areas is None:
            all_areas = set(self.session_df['area'].dropna().unique())
            areas = [a for a in ['V1', 'LM', 'RL', 'AM', 'PM', 'AL'] if a in all_areas]

        conditions = ['monocular', 'binocular']

        # Bootstrap per area per condition
        area_stats = {}
        for area in areas:
            area_df = self.filter_by_area(self.session_df, area_name=area)
            reliable_df = self.filter_by_reliability(area_df, reliable_type='reliable', threshold=0.85)
            n = len(reliable_df)
            if n <= 100:
                area_stats[area] = None
                continue
            cond_stats = {}
            for cond in conditions:
                boot_results = []
                for _ in range(n_bootstrap):
                    sample = reliable_df.sample(n=n, replace=True)
                    boot_results.append(
                        len(self.filter_by_modulation(sample, modulation_type=cond)) / n * 100
                    )
                n_mod_obs = len(self.filter_by_modulation(reliable_df, modulation_type=cond))
                cond_stats[cond] = {
                    'mean': n_mod_obs / n * 100,
                    'ci_low': np.percentile(boot_results, 2.5),
                    'ci_high': np.percentile(boot_results, 97.5),
                }
            area_stats[area] = {'conditions': cond_stats, 'n': n}

        areas = [a for a in areas if area_stats.get(a) is not None]
        n_areas = len(areas)
        n_conds = len(conditions)
        bar_width = 0.45
        bar_gap = 0.05
        x = np.arange(n_areas) * 1.4

        fig, ax = plt.subplots(figsize=(max(4.5, n_areas *1.2  + 2), 4.5))

        for i, cond in enumerate(conditions):
            offset = (i - n_conds / 2 + 0.5) * (bar_width + bar_gap)
            for j, area in enumerate(areas):
                s = area_stats[area]['conditions'][cond]
                mean = s['mean']
                ax.bar(x[j] + offset, mean, width=bar_width,
                       color=self.modulation_colors[cond],
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=cond.capitalize() if j == 0 else '')
                ax.errorbar(x[j] + offset, mean,
                            yerr=[[mean - s['ci_low']], [s['ci_high'] - mean]],
                            fmt='none', color='black', capsize=3, linewidth=1.5)

        for j, area in enumerate(areas):
            n = area_stats[area]['n']
            top = max(area_stats[area]['conditions'][c]['ci_high'] for c in conditions)
            ax.text(x[j], top + 0.3, f'n={n}',
                    ha='center', va='bottom', fontsize=10, color='gray')

        ax.set_ylabel('% of Reliable Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(areas, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        fig.suptitle('Modulated Cells per Area', fontsize=20, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)
        plt.tight_layout()
        return fig

    def plot_distance_per_area(self, condition='binocular', n_bootstrap=1000, areas=None):
        """
        Preferred distance distribution per area with 95% bootstrap CI.

        Pools all animals together per area and bootstraps from the whole pool
        (n = total modulated cells in that area across all animals).

        Parameters
        ----------
        condition : str
            'monocular', 'binocular', or 'both'. Default 'binocular'.
        n_bootstrap : int
            Bootstrap iterations. Default 1000.
        areas : list of str, optional
            Areas to include. Default ['V1', 'LM', 'RL'].
        """
        if areas is None:
            all_areas = set(self.session_df['area'].dropna().unique())
            areas = [a for a in ['V1', 'LM', 'RL', 'AM', 'PM', 'AL'] if a in all_areas]

        distances = [0, 1, 2, 3]
        dist_labels = ['10cm', '16cm', '22cm', '28cm']

        def _get_pref_dists(mod_df):
            if condition == 'monocular':
                return mod_df['monocular'].apply(lambda x: x['distance']).values.astype(float)
            elif condition == 'binocular':
                return mod_df['binocular'].apply(lambda x: x['distance']).values.astype(float)
            else:
                mono = mod_df['monocular'].apply(lambda x: x['distance']).values.astype(float)
                bino = mod_df['binocular'].apply(lambda x: x['distance']).values.astype(float)
                return (mono + bino) / 2

        area_stats = {}
        for area in areas:
            area_df = self.filter_by_area(self.session_df, area_name=area)
            mod_df = self.filter_by_modulation(area_df, modulation_type=condition)
            n = len(mod_df)
            if n <= 50:
                area_stats[area] = None
                continue
            pref = _get_pref_dists(mod_df)
            obs_rounded = [int(np.round(v)) for v in pref]
            boot = {d: [] for d in distances}
            for _ in range(n_bootstrap):
                sample = mod_df.sample(n=n, replace=True)
                sample_pref = _get_pref_dists(sample)
                rounded = [int(np.round(v)) for v in sample_pref]
                for d in distances:
                    boot[d].append(np.sum(np.array(rounded) == d) / n * 100)
            area_stats[area] = {
                d: {
                    'mean': np.sum(np.array(obs_rounded) == d) / n * 100,
                    'ci_low': np.percentile(boot[d], 2.5),
                    'ci_high': np.percentile(boot[d], 97.5),
                }
                for d in distances
            }
            area_stats[area]['n'] = n

        areas = [a for a in areas if area_stats.get(a) is not None]
        n_areas = len(areas)
        n_dists = len(distances)
        bar_width = 0.18
        bar_gap = 0.02
        group_width = n_dists * (bar_width + bar_gap)
        x = np.arange(n_areas) * (group_width + 0.4)

        fig, ax = plt.subplots(figsize=(max(8, n_areas * (group_width + 0.6) + 2), 4.5))

        for j, area in enumerate(areas):
            s = area_stats[area]
            for k, dist in enumerate(distances):
                offset = (k - n_dists / 2 + 0.5) * (bar_width + bar_gap)
                mean = s[dist]['mean']
                yerr_low = mean - s[dist]['ci_low']
                yerr_high = s[dist]['ci_high'] - mean
                ax.bar(x[j] + offset, mean, width=bar_width,
                       color=self.distance_colors[k], edgecolor='black', linewidth=1.0, alpha=0.85,
                       label=dist_labels[k] if j == 0 else '')
                ax.errorbar(x[j] + offset, mean, yerr=[[yerr_low], [yerr_high]],
                            fmt='none', color='black', capsize=2, linewidth=1.2)
            ax.text(x[j], max(s[d]['ci_high'] for d in distances) + 0.5,
                    f'n={s["n"]}', ha='center', va='bottom', fontsize=10, color='gray')

        ax.set_ylabel('% of Modulated Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(areas, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        condition_label = condition.capitalize() if condition != 'both' else 'Combined'
        fig.suptitle(f'Preferred Distance per Area ({condition_label})', fontsize=20, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)
        plt.tight_layout()
        return fig

    def plot_modulation_bars(self, group_by=None, min_cells=50):
        """
        Bar plot of % modulated cells per group, with SEM error bars and per-animal scatter dots.

        Mirrors MultiSessionVisualizer.plot_modulation_bars but treats each animal as a replicate:
        bar height = mean across animals, error bars = SEM across animals, dots = per-animal values.

        Parameters
        ----------
        group_by : str or None
            'area', 'session', or None (all cells together).
        min_cells : int
            Minimum number of reliable cells an animal must have in a group to be included (default: 10).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if group_by == 'area':
            groups = self.get_available_areas()
        elif group_by == 'session':
            groups = self.get_available_sessions()
        elif group_by is None:
            groups = ['all_cells']
        else:
            raise ValueError(f"group_by must be 'area', 'session', or None, got {group_by}")

        animals = sorted(self.session_df['animal'].unique())
        modulation_types = ['monocular', 'binocular']

        # Per-animal, per-group, per-modulation-type values
        animal_vals = {g: {m: [] for m in modulation_types} for g in groups}
        for animal in animals:
            animal_df = self.session_df[self.session_df['animal'] == animal]
            for group in groups:
                if group_by == 'area':
                    group_df = self.filter_by_area(animal_df, area_name=group)
                elif group_by == 'session':
                    group_df = self.filter_by_session(animal_df, session_name=group)
                else:
                    group_df = animal_df
                reliable_df = self.filter_by_reliability(group_df, reliable_type='reliable', threshold=0.85)
                n_reliable = len(reliable_df)
                for mod_type in modulation_types:
                    if n_reliable >= min_cells:
                        pct = len(self.filter_by_modulation(reliable_df, modulation_type=mod_type)) / n_reliable * 100
                    else:
                        pct = np.nan
                    animal_vals[group][mod_type].append(pct)

        # Filter out groups where fewer than 2 animals have sufficient cells
        groups = [g for g in groups if np.sum(~np.isnan(animal_vals[g]['binocular'])) >= 2]

        n_groups = len(groups)
        n_mod = len(modulation_types)
        x = np.arange(n_groups) * 1.2
        bar_width = 0.45
        bar_gap = 0.03

        animal_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        marker_map = {a: animal_markers[k % len(animal_markers)] for k, a in enumerate(animals)}

        rng = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=(max(8, n_groups * 2), 7))

        for j, mod_type in enumerate(modulation_types):
            offset = (j - n_mod / 2 + 0.5) * (bar_width + bar_gap)
            for i, group in enumerate(groups):
                vals = np.array(animal_vals[group][mod_type])
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                mean = np.mean(vals[valid])
                sem = np.std(vals[valid], ddof=1) / np.sqrt(valid.sum()) if valid.sum() > 1 else 0

                ax.bar(x[i] + offset, mean, width=bar_width,
                       color=self.modulation_colors[mod_type],
                       edgecolor='black', linewidth=0.5, alpha=0.85,
                       label=mod_type.capitalize() if i == 0 else '')
                ax.errorbar(x[i] + offset, mean, yerr=sem,
                            fmt='none', color='black', capsize=4, linewidth=1.5)
                for animal, val in zip(animals, vals):
                    if not np.isnan(val):
                        marker = marker_map[animal]
                        jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3)
                        ax.scatter(x[i] + offset + jitter, val, marker=marker, color='#888888',
                                   s=50, zorder=5, alpha=0.6, edgecolors='white', linewidths=0.5,
                                   label=animal if (i == 0 and j == 0) else '')

        ax.set_ylabel('Percentage of Reliable Cells (%)', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        from matplotlib.lines import Line2D
        animal_handles = [Line2D([0], [0], marker=marker_map[a], color='w',
                                 markerfacecolor='#888888', markeredgecolor='white',
                                 markersize=8, label=a) for a in animals]
        ax.legend(handles=animal_handles, loc='upper right',
                  framealpha=0.9, edgecolor='gray', fontsize=10, title='Animal')

        plt.tight_layout()
        fig.suptitle('Modulated Cells', fontsize=22, color='black', y=1.05, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        return fig

    def plot_distance_bars(self, group_by=None, condition='both', min_cells=10, rf_range=None):
        """
        Bar plot of preferred distance distribution per group, with SEM error bars and per-animal scatter dots.

        Mirrors MultiSessionVisualizer.plot_distance_bars but treats each animal as a replicate:
        bar height = mean across animals, error bars = SEM across animals, dots = per-animal values.

        Parameters
        ----------
        group_by : str or None
            'area', 'session', or None (all cells together).
        condition : str
            'monocular', 'binocular', 'both', 'reliable', or 'un-modulated'.
        min_cells : int
            Minimum number of target cells an animal must have in a group to be included (default: 10).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if group_by == 'area':
            groups = self.get_available_areas()
        elif group_by == 'session':
            groups = self.get_available_sessions()
        elif group_by is None:
            groups = ['all_cells']
        else:
            raise ValueError(f"group_by must be 'area', 'session', or None, got {group_by}")

        animals = sorted(self.session_df['animal'].unique())
        distances = [0, 1, 2, 3]

        def _extract_pref_dists(df, cond):
            if cond == 'reliable':
                return df['distance'].values.astype(float)
            elif cond == 'un-modulated':
                return df['distance'].values.astype(float)
            elif cond == 'monocular':
                return df['monocular'].apply(lambda x: x['distance']).values.astype(float)
            elif cond == 'binocular':
                return df['binocular'].apply(lambda x: x['distance']).values.astype(float)
            else:  # 'both'
                mono = df['monocular'].apply(lambda x: x['distance']).values.astype(float)
                bino = df['binocular'].apply(lambda x: x['distance']).values.astype(float)
                return (mono + bino) / 2

        # Per-animal, per-group, per-distance values
        animal_vals = {g: {d: [] for d in distances} for g in groups}
        for animal in animals:
            animal_df = self.session_df[self.session_df['animal'] == animal]
            for group in groups:
                if group_by == 'area':
                    group_df = self.filter_by_area(animal_df, area_name=group)
                elif group_by == 'session':
                    group_df = self.filter_by_session(animal_df, session_name=group)
                else:
                    group_df = animal_df

                if condition == 'reliable':
                    target_df = self.filter_by_reliability(group_df, reliable_type='reliable', threshold=0.85)
                elif condition == 'un-modulated':
                    target_df = self.filter_by_modulation(group_df, modulation_type='none')
                else:
                    target_df = self.filter_by_modulation(group_df, modulation_type=condition)

                target_df = self._filter_by_rf_range(target_df, condition, rf_range)
                n_target = len(target_df)
                if n_target >= min_cells:
                    pref = _extract_pref_dists(target_df, condition)
                    rounded = [int(np.round(v)) for v in pref]
                    for dist in distances:
                        animal_vals[group][dist].append(np.sum(np.array(rounded) == dist) / n_target * 100)
                else:
                    for dist in distances:
                        animal_vals[group][dist].append(np.nan)

        # Filter out groups where fewer than 2 animals have sufficient cells
        groups = [g for g in groups if np.sum(~np.isnan(animal_vals[g][0])) >= 2]

        n_groups = len(groups)
        n_dists = len(distances)
        bar_width = 0.27
        bar_gap = 0.02
        x = np.arange(n_groups) * 1.4

        animal_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        marker_map = {a: animal_markers[k % len(animal_markers)] for k, a in enumerate(animals)}

        rng = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=(max(8, n_groups * 2), 7))

        for j, dist in enumerate(distances):
            offset = (j - n_dists / 2 + 0.5) * (bar_width + bar_gap)
            for i, group in enumerate(groups):
                vals = np.array(animal_vals[group][dist])
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                mean = np.mean(vals[valid])
                sem = np.std(vals[valid], ddof=1) / np.sqrt(valid.sum()) if valid.sum() > 1 else 0

                ax.bar(x[i] + offset, mean, width=bar_width,
                       color=self.distance_colors[j],
                       edgecolor='black', linewidth=0.5, alpha=0.85,
                       label=f'Distance {dist}' if i == 0 else '')
                ax.errorbar(x[i] + offset, mean, yerr=sem,
                            fmt='none', color='black', capsize=3, linewidth=1.5)
                for animal, val in zip(animals, vals):
                    if not np.isnan(val):
                        marker = marker_map[animal]
                        jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3)
                        ax.scatter(x[i] + offset + jitter, val, marker=marker, color='#888888',
                                   s=50, zorder=5, alpha=0.6, edgecolors='white', linewidths=0.5,
                                   label=animal if (i == 0 and j == 0) else '')

        if condition in ('reliable', 'un-modulated'):
            ax.set_ylabel('Percentage of Reliable Cells (%)', fontsize=16)
        else:
            ax.set_ylabel('Percentage of Modulated Cells (%)', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        from matplotlib.lines import Line2D
        animal_handles = [Line2D([0], [0], marker=marker_map[a], color='w',
                                 markerfacecolor='#888888', markeredgecolor='white',
                                 markersize=8, label=a) for a in animals]
        ax.legend(handles=animal_handles, loc='upper right',
                  framealpha=0.9, edgecolor='gray', fontsize=10, title='Animal')

        plt.tight_layout()
        condition_label = condition.capitalize() if condition != 'both' else 'Combined'
        fig.suptitle(f'Preferred Distance ({condition_label})', fontsize=22, color='black', y=1.05, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        return fig

    def plot_distance_pref_vs_rf(self, condition='both', area=None,
                                 x_axis='RF', bin_width=10.0,
                                 min_cells_per_bin=20,
                                 axis_range=None, ax=None):
        """
        Per-animal distribution of preferred distance across bins of `x_axis`,
        with SEM bars across animals.

        Each animal contributes one curve per distance (% of its cells in each
        bin preferring that distance). The plotted line is the mean across
        animals; vertical error bars show +/- SEM between animals. Bins where
        an animal has fewer than `min_cells_per_bin` cells contribute NaN, and
        bins with < 2 contributing animals are masked out.

        For `x_axis` in ('azi', 'alt'), values are linearly rescaled per-animal
        to the canonical range from `_AXIS_LIMITS`. This way each animal's
        widefield range maps to the same display range, making cross-animal
        comparison meaningful even with different fields of view.

        Parameters
        ----------
        condition : str
            'monocular', 'binocular', 'both', 'reliable', or 'un-modulated'.
        area : str, optional
            Filter cells by visual area (e.g. 'V1').
        x_axis : str
            'RF', 'azi', or 'alt'.
        bin_width : float
            Bin width in degrees (default 10.0).
        min_cells_per_bin : int
            Per-animal minimum cell count per bin (default 20).
        axis_range : tuple of (float, float), optional
            (min, max) range for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        df = self.session_df
        if area is not None:
            df = self.filter_by_area(df, area_name=area)

        animals = sorted(df['animal'].unique())
        distances = [0, 1, 2, 3]

        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        n_bins = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        animal_pct = np.full((len(animals), n_bins, len(distances)), np.nan)

        for ai, animal in enumerate(animals):
            animal_df = df[df['animal'] == animal]

            if condition == 'reliable':
                target_df = self.filter_by_reliability(animal_df, reliable_type='reliable', threshold=0.85)
                dist_vals = self.extract_general_feature(target_df, feature='distance')
            elif condition == 'un-modulated':
                target_df = self.filter_by_modulation(animal_df, modulation_type='none')
                dist_vals = self.extract_general_feature(target_df, feature='distance')
            else:
                target_df = self.filter_by_modulation(animal_df, modulation_type=condition)
                dist_vals = self.extract_condition_feature(target_df, condition=condition, feature='distance')

            x_vals = self._get_x_axis_values(target_df, x_axis, condition)

            x_vals = np.asarray(x_vals, dtype=float)
            dist_vals = np.asarray(dist_vals, dtype=float)
            valid = ~(np.isnan(x_vals) | np.isnan(dist_vals))
            x_vals = x_vals[valid]
            dist_vals = np.round(dist_vals[valid]).astype(int)

            if len(x_vals) == 0:
                continue

            bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
            counts = (pd.crosstab(bin_idx, dist_vals)
                      .reindex(index=range(n_bins), columns=distances, fill_value=0))
            totals = counts.sum(axis=1)
            pct = counts.div(totals.replace(0, np.nan), axis=0) * 100
            pct.loc[totals < min_cells_per_bin, :] = np.nan
            animal_pct[ai] = pct.values

        n_per_bin = np.sum(~np.isnan(animal_pct[:, :, 0]), axis=0)
        bin_has_data = n_per_bin >= 2

        mean = np.full((n_bins, len(distances)), np.nan)
        sem = np.full((n_bins, len(distances)), np.nan)
        if bin_has_data.any():
            with np.errstate(invalid='ignore'):
                mean[bin_has_data] = np.nanmean(animal_pct[:, bin_has_data, :], axis=0)
                sem[bin_has_data] = (np.nanstd(animal_pct[:, bin_has_data, :], axis=0, ddof=1)
                                     / np.sqrt(n_per_bin[bin_has_data, None]))

        if not bin_has_data.any():
            ax.set_title(f'No bins with >= 2 animals at >= {min_cells_per_bin} cells', fontsize=16)
            return fig, ax

        for j, dist in enumerate(distances):
            m = mean[:, j]
            s = sem[:, j]
            ax.errorbar(centers, m, yerr=s, marker='o', linewidth=2,
                        capsize=3, color=self.distance_colors[j],
                        label=f'Distance {dist}')

        for k in range(n_bins):
            if bin_has_data[k]:
                top = np.nanmax(mean[k] + sem[k])
                ax.text(centers[k], top + 2, f'n={int(n_per_bin[k])}',
                        ha='center', va='bottom', fontsize=10, color='black')

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        ax.set_ylabel('Percentage of Cells (%)', fontsize=16)
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        condition_label = {'reliable': 'Reliable', 'un-modulated': 'Un-modulated',
                           'both': 'Combined'}.get(condition, condition.capitalize())
        title_parts = [condition_label]
        if area is not None:
            title_parts.append(area)
        fig.suptitle(f'Preferred Distance vs {x_axis} ({" | ".join(title_parts)})',
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

    def plot_distance_pref_vs_rf_compare(self, groups,
                                         x_axis='RF', bin_width=10.0,
                                         min_cells_per_bin=20,
                                         axis_range=None, rf_range=None,
                                         pct_of_distance=None,
                                         show_stats=True, ax=None):
        """
        Mean continuous preferred distance vs `x_axis`, one line per
        (area, condition) group, pooled across all animals. Error bars are
        SEM across cells in each bin (`std(ddof=1) / sqrt(n)`).

        Use for cross-group comparison (e.g. V1 vs RL; V1-binocular vs
        V1-monocular). No per-animal split — all cells in a group are pooled.

        Parameters
        ----------
        groups : list of (str or None, str)
            (area, condition) tuples. `area=None` means no area filter.
            `condition` is 'monocular', 'binocular', 'both', 'reliable', or
            'un-modulated'.
        x_axis : str
            'RF', 'azi', or 'alt'.
        bin_width : float
            Bin width in degrees.
        min_cells_per_bin : int
            Bins below this cell count become NaN (line breaks).
        axis_range : tuple of (float, float), optional
            (min, max) for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        rf_range : tuple of (float, float), optional
            Condition-aware RF window filter.
        pct_of_distance : int or None
            If None (default), y is mean continuous distance per bin (0–3
            axis). If 0/1/2/3, y becomes the % of cells in each bin whose
            (rounded) preferred distance equals this value (0–100 axis);
            error bars are binomial SEM `sqrt(p*(1-p)/n) * 100`. Per-bin
            stats switch from Mann-Whitney U to Fisher's exact.
        show_stats : bool
            If True and >= 2 groups have data, runs a per-bin test on the
            first two groups (Mann-Whitney U in mean mode; Fisher's exact
            in pct mode) — only on bins with >= `min_cells_per_bin` cells
            in both — corrects p-values across bins with Benjamini-Hochberg
            (alpha=0.05), and marks surviving bins with a star above the
            point. Descriptive only — does not constitute a single global
            claim about the curves.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        n_bins = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        area_color_map = self._get_area_colors('area')
        fallback_cycle = list(plt.cm.tab10.colors)
        category_colors = {'binocular': 'tab:blue', 'monocular': 'tab:red',
                           'un-modulated': 'gray', 'none': 'gray'}
        color_by_category = len({a for (a, _) in groups}) == 1
        any_plotted = False
        group_data = []

        for gi, (area, condition) in enumerate(groups):
            df = self.session_df
            if area is not None:
                df = self.filter_by_area(df, area_name=area)

            if condition == 'reliable':
                target_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
            elif condition == 'un-modulated':
                target_df = self.filter_by_modulation(df, modulation_type='none')
            else:
                target_df = self.filter_by_modulation(df, modulation_type=condition)

            target_df = self._filter_by_rf_range(target_df, condition, rf_range)

            if condition in ('reliable', 'un-modulated'):
                dist_vals = self.extract_general_feature(target_df, feature='distance')
            else:
                dist_vals = self.extract_condition_feature(target_df, condition=condition, feature='distance')

            x_vals = self._get_x_axis_values(target_df, x_axis, condition)

            x_vals = np.asarray(x_vals, dtype=float)
            dist_vals = np.asarray(dist_vals, dtype=float)
            valid = ~(np.isnan(x_vals) | np.isnan(dist_vals))
            x_vals = x_vals[valid]
            dist_vals = dist_vals[valid]

            if len(x_vals) == 0:
                group_data.append(None)
                continue

            bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
            bin_idx_arr = np.asarray(bin_idx, dtype=float)

            if pct_of_distance is None:
                df_tmp = pd.DataFrame({'bin': bin_idx, 'dist': dist_vals})
                grouped = df_tmp.groupby('bin')['dist']
                means = grouped.mean().reindex(range(n_bins)).values
                stds = grouped.std(ddof=1).reindex(range(n_bins)).values
                n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
                with np.errstate(invalid='ignore', divide='ignore'):
                    sems = stds / np.sqrt(n_per)
            else:
                is_d = (np.round(dist_vals).astype(int) == pct_of_distance).astype(float)
                df_tmp = pd.DataFrame({'bin': bin_idx, 'is_d': is_d})
                grouped = df_tmp.groupby('bin')['is_d']
                p_hat = grouped.mean().reindex(range(n_bins)).values
                n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
                means = p_hat * 100
                with np.errstate(invalid='ignore', divide='ignore'):
                    sems = np.sqrt(p_hat * (1 - p_hat) / n_per) * 100

            mask = n_per < min_cells_per_bin
            means[mask] = np.nan
            sems[mask] = np.nan

            if color_by_category:
                color = category_colors.get(condition, fallback_cycle[gi % len(fallback_cycle)])
            else:
                color = area_color_map.get(area) if area is not None else None
                if color is None:
                    color = fallback_cycle[gi % len(fallback_cycle)]

            if color_by_category and pct_of_distance is not None and condition in ('monocular', 'binocular', 'both'):
                color = self.distance_colors[pct_of_distance]

            label = f'{area if area is not None else "any"} {condition}'

            ax.errorbar(centers, means, yerr=sems, marker='o', linewidth=2,
                        capsize=3, color=color, label=label)
            any_plotted = True

            bin_dist = {k: dist_vals[bin_idx_arr == k] for k in range(n_bins)}
            group_data.append({'means': means, 'sems': sems,
                               'n_per': n_per, 'bin_dist': bin_dist})

        if not any_plotted:
            ax.set_title('No cells for any group', fontsize=16)
            return fig, ax

        if show_stats:
            valid_groups = [g for g in group_data if g is not None]
            if len(valid_groups) >= 2:
                g1, g2 = valid_groups[0], valid_groups[1]
                pvals = np.full(n_bins, np.nan)
                for k in range(n_bins):
                    if (g1['n_per'][k] >= min_cells_per_bin and
                            g2['n_per'][k] >= min_cells_per_bin):
                        a = g1['bin_dist'][k]
                        b = g2['bin_dist'][k]
                        try:
                            if pct_of_distance is None:
                                _, p = mannwhitneyu(a, b, alternative='two-sided')
                            else:
                                a_d = int(np.sum(np.round(a).astype(int) == pct_of_distance))
                                b_d = int(np.sum(np.round(b).astype(int) == pct_of_distance))
                                table = [[a_d, len(a) - a_d],
                                         [b_d, len(b) - b_d]]
                                _, p = fisher_exact(table, alternative='two-sided')
                            pvals[k] = p
                        except ValueError:
                            pass

                tested = ~np.isnan(pvals)
                if tested.any():
                    p_sub = pvals[tested]
                    n_tests = len(p_sub)
                    sorted_p = np.sort(p_sub)
                    thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
                    under = sorted_p <= thresholds
                    cutoff = sorted_p[np.where(under)[0].max()] if under.any() else -np.inf
                    sig_sub = p_sub <= cutoff

                    sig = np.zeros(n_bins, dtype=bool)
                    sig[tested] = sig_sub

                    for k in range(n_bins):
                        if sig[k]:
                            tops = []
                            for g in (g1, g2):
                                m = g['means'][k]
                                s = g['sems'][k] if not np.isnan(g['sems'][k]) else 0.0
                                if not np.isnan(m):
                                    tops.append(m + s)
                            if tops:
                                star_offset = 0.1 if pct_of_distance is None else 5.0
                                ax.text(centers[k], max(tops) + star_offset, '*',
                                        ha='center', va='bottom',
                                        fontsize=18, color='black')

        y_max_curves = 0.0
        for g in group_data:
            if g is None:
                continue
            sems_safe = np.where(np.isnan(g['sems']), 0.0, g['sems'])
            vals = g['means'] + sems_safe
            if not np.all(np.isnan(vals)):
                y_max_curves = max(y_max_curves, float(np.nanmax(vals)))

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        if pct_of_distance is None:
            ax.set_ylabel('Mean Distance', fontsize=16)
            ax.set_ylim(0, min(3, y_max_curves + 0.2))
            suptitle = f'Mean Distance vs {x_axis} (comparison)'
        else:
            ax.set_ylabel(f'% at Distance {pct_of_distance}', fontsize=16)
            ax.set_ylim(0, min(100, y_max_curves + 10.0))
            suptitle = f'% at Distance {pct_of_distance} vs {x_axis} (comparison)'
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        fig.suptitle(suptitle,
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

    def plot_modulation_by_rf_compare(self, groups,
                                      x_axis='RF', bin_width=10.0,
                                      min_cells_per_bin=20,
                                      axis_range=None, rf_range=None,
                                      show_stats=True, ax=None):
        """
        % of reliable cells of a given modulation category vs `x_axis`,
        one line per (area, modulation_type) group, pooled across all
        animals. Error bars are binomial SEM `sqrt(p*(1-p)/n) * 100`.

        Denominator per bin is all reliable cells in that (area, RF) bin;
        numerator is the subset matching `modulation_type`.

        Parameters
        ----------
        groups : list of (str or None, str)
            (area, modulation_type) tuples. `area=None` means no area
            filter. `modulation_type` is 'monocular', 'binocular', 'both',
            or 'un-modulated' (treated as 'none').
        x_axis : str
            'RF', 'azi', or 'alt'.
        bin_width : float
            Bin width in degrees (default 10.0).
        min_cells_per_bin : int
            Bins below this reliable-cell count become NaN (line breaks).
        axis_range : tuple of (float, float), optional
            (min, max) for bin edges. Defaults to `_AXIS_LIMITS[x_axis]`.
        rf_range : tuple of (float, float), optional
            RF window filter applied to the reliable pool (general RF).
        show_stats : bool
            If True and >= 2 groups have data, runs a per-bin Fisher's
            exact test on the first two groups — only on bins with >=
            `min_cells_per_bin` reliable cells in both — corrects across
            bins with Benjamini-Hochberg (alpha=0.05), and marks
            surviving bins with a star above the point. Descriptive only.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        if axis_range is None:
            x_min, x_max = self._AXIS_LIMITS[x_axis]
        else:
            x_min, x_max = axis_range

        edges = np.arange(x_min, x_max + bin_width, bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        n_bins = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        area_color_map = self._get_area_colors('area')
        fallback_cycle = list(plt.cm.tab10.colors)
        category_colors = {'binocular': 'tab:blue', 'monocular': 'tab:red',
                           'un-modulated': 'gray', 'none': 'gray'}
        color_by_category = len({a for (a, _) in groups}) == 1
        any_plotted = False
        group_data = []

        for gi, (area, modulation_type) in enumerate(groups):
            df = self.session_df
            if area is not None:
                df = self.filter_by_area(df, area_name=area)

            reliable_df = self.filter_by_reliability(df, reliable_type='reliable', threshold=0.85)
            reliable_df = self._filter_by_rf_range(reliable_df, 'reliable', rf_range)

            mod_key = 'none' if modulation_type == 'un-modulated' else modulation_type
            mod_df = self.filter_by_modulation(reliable_df, modulation_type=mod_key)

            x_vals = self._get_x_axis_values(reliable_df, x_axis, 'reliable')
            x_vals = np.asarray(x_vals, dtype=float)
            is_mod = reliable_df.index.isin(mod_df.index).astype(float)

            valid = ~np.isnan(x_vals)
            x_vals = x_vals[valid]
            is_mod = is_mod[valid]

            if len(x_vals) == 0:
                group_data.append(None)
                continue

            bin_idx = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
            bin_idx_arr = np.asarray(bin_idx, dtype=float)

            df_tmp = pd.DataFrame({'bin': bin_idx, 'is_mod': is_mod})
            grouped = df_tmp.groupby('bin')['is_mod']
            p_hat = grouped.mean().reindex(range(n_bins)).values
            n_per = grouped.count().reindex(range(n_bins), fill_value=0).values
            means = p_hat * 100
            with np.errstate(invalid='ignore', divide='ignore'):
                sems = np.sqrt(p_hat * (1 - p_hat) / n_per) * 100

            mask = n_per < min_cells_per_bin
            means[mask] = np.nan
            sems[mask] = np.nan

            if color_by_category:
                color = category_colors.get(modulation_type, fallback_cycle[gi % len(fallback_cycle)])
            else:
                color = area_color_map.get(area) if area is not None else None
                if color is None:
                    color = fallback_cycle[gi % len(fallback_cycle)]

            label = f'{area if area is not None else "any"} {modulation_type}'

            ax.errorbar(centers, means, yerr=sems, marker='o', linewidth=2,
                        capsize=3, color=color, label=label)
            any_plotted = True

            bin_mod = {k: is_mod[bin_idx_arr == k] for k in range(n_bins)}
            group_data.append({'means': means, 'sems': sems,
                               'n_per': n_per, 'bin_mod': bin_mod})

        if not any_plotted:
            ax.set_title('No cells for any group', fontsize=16)
            return fig, ax

        if show_stats:
            valid_groups = [g for g in group_data if g is not None]
            if len(valid_groups) >= 2:
                g1, g2 = valid_groups[0], valid_groups[1]
                pvals = np.full(n_bins, np.nan)
                for k in range(n_bins):
                    if (g1['n_per'][k] >= min_cells_per_bin and
                            g2['n_per'][k] >= min_cells_per_bin):
                        a = g1['bin_mod'][k]
                        b = g2['bin_mod'][k]
                        try:
                            a_m = int(np.sum(a == 1))
                            b_m = int(np.sum(b == 1))
                            table = [[a_m, len(a) - a_m],
                                     [b_m, len(b) - b_m]]
                            _, p = fisher_exact(table, alternative='two-sided')
                            pvals[k] = p
                        except ValueError:
                            pass

                tested = ~np.isnan(pvals)
                if tested.any():
                    p_sub = pvals[tested]
                    n_tests = len(p_sub)
                    sorted_p = np.sort(p_sub)
                    thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
                    under = sorted_p <= thresholds
                    cutoff = sorted_p[np.where(under)[0].max()] if under.any() else -np.inf
                    sig_sub = p_sub <= cutoff

                    sig = np.zeros(n_bins, dtype=bool)
                    sig[tested] = sig_sub

                    for k in range(n_bins):
                        if sig[k]:
                            tops = []
                            for g in (g1, g2):
                                m = g['means'][k]
                                s = g['sems'][k] if not np.isnan(g['sems'][k]) else 0.0
                                if not np.isnan(m):
                                    tops.append(m + s)
                            if tops:
                                ax.text(centers[k], max(tops) + 5.0, '*',
                                        ha='center', va='bottom',
                                        fontsize=18, color='black')

        y_max_curves = 0.0
        for g in group_data:
            if g is None:
                continue
            sems_safe = np.where(np.isnan(g['sems']), 0.0, g['sems'])
            vals = g['means'] + sems_safe
            if not np.all(np.isnan(vals)):
                y_max_curves = max(y_max_curves, float(np.nanmax(vals)))

        ax.set_xlabel(f'{x_axis} (deg)', fontsize=16)
        ax.set_ylabel('% of Reliable Cells', fontsize=16)
        ax.set_ylim(0, min(100, y_max_curves + 10.0))
        ax.set_xlim(*self._AXIS_LIMITS[x_axis])
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)

        fig.suptitle(f'% Modulated Cells vs {x_axis} (comparison)',
                     fontsize=20, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=16, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

    def plot_distance_boxplot(self, condition='binocular',
                              areas=('V1', 'LM', 'RL'),
                              rf_range=None, show_stats=True, ax=None):
        """
        Boxplot of preferred distance per visual area, pooling all cells from
        all animals as if it were one session.

        Each dot is one cell. Box = quartiles, line = median, whiskers per
        matplotlib default. No significance brackets.

        Parameters
        ----------
        condition : str
            'monocular', 'binocular' (default), 'both', 'reliable', or 'un-modulated'.
            Selects which cells contribute and which `distance` value to extract.
        areas : tuple of str
            Areas to compare, left-to-right (default V1, LM, RL).
        rf_range : tuple of (float, float), optional
            (min, max) RF window. Cells outside are excluded.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig, ax
        """
        distance_per_area = {}
        for area in areas:
            area_df = self.filter_by_area(self.session_df, area_name=area)

            if condition == 'reliable':
                target_df = self.filter_by_reliability(area_df, reliable_type='reliable', threshold=0.85)
            elif condition == 'un-modulated':
                target_df = self.filter_by_modulation(area_df, modulation_type='none')
            else:
                target_df = self.filter_by_modulation(area_df, modulation_type=condition)

            target_df = self._filter_by_rf_range(target_df, condition, rf_range)

            if condition in ('reliable', 'un-modulated'):
                dist_vals = self.extract_general_feature(target_df, feature='distance')
            else:
                dist_vals = self.extract_condition_feature(target_df, condition=condition, feature='distance')

            dist_vals = np.asarray(dist_vals, dtype=float)
            dist_vals = dist_vals[~np.isnan(dist_vals)]
            distance_per_area[area] = dist_vals

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, len(areas) * 1.5), 6))
        else:
            fig = ax.get_figure()

        data = [distance_per_area[a] for a in areas]

        area_color_map = self._get_area_colors('area')
        default_cycle = ['#4477AA', '#228833', '#EE6677', '#CCBB44', '#66CCEE', '#AA3377']
        colors = [area_color_map.get(a, default_cycle[i % len(default_cycle)])
                  for i, a in enumerate(areas)]

        non_empty = [i for i, d in enumerate(data) if len(d) > 0]
        if not non_empty:
            ax.set_title('No cells in any of the selected areas', fontsize=16)
            return fig, ax

        bp = ax.boxplot([data[i] for i in non_empty],
                        positions=non_empty, widths=0.5,
                        showfliers=False, patch_artist=True)

        for k, i in enumerate(non_empty):
            color = colors[i]
            bp['boxes'][k].set_facecolor('white')
            bp['boxes'][k].set_edgecolor(color)
            bp['boxes'][k].set_linewidth(2)
            bp['medians'][k].set_color(color)
            bp['medians'][k].set_linewidth(2)
            for w in (bp['whiskers'][k * 2], bp['whiskers'][k * 2 + 1]):
                w.set_color(color)
            for c in (bp['caps'][k * 2], bp['caps'][k * 2 + 1]):
                c.set_color(color)

        rng = np.random.default_rng(42)
        for i, area in enumerate(areas):
            vals = data[i]
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=colors[i], alpha=0.2, s=10,
                       edgecolor='none', zorder=3)

        if show_stats and len(non_empty) >= 2:
            pairs = list(combinations(non_empty, 2))
            n_comp = len(pairs)
            y_max = max(np.max(data[i]) for i in non_empty)
            y_min = min(np.min(data[i]) for i in non_empty)
            step = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
            tick_len = step * 0.25

            for k, (i, j) in enumerate(pairs):
                _, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
                p_adj = min(p * n_comp, 1.0)
                if p_adj < 0.001:
                    label = '***'
                elif p_adj < 0.01:
                    label = '**'
                elif p_adj < 0.05:
                    label = '*'
                else:
                    label = 'n.s.'

                y = y_max + step * (k + 1)
                ax.plot([i, i, j, j], [y - tick_len, y, y, y - tick_len],
                        color='black', linewidth=1.2)
                ax.text((i + j) / 2, y + tick_len * 0.4, label,
                        ha='center', va='bottom', fontsize=12,
                        fontweight='bold' if label != 'n.s.' else 'normal')

            ax.set_ylim(top=y_max + step * (n_comp + 1.5))

        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels([f'{a}\nn={len(distance_per_area[a])}' for a in areas],
                           fontsize=14)
        ax.set_ylabel(f'Distance ({condition})', fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        condition_label = {'reliable': 'Reliable', 'un-modulated': 'Un-modulated',
                           'both': 'Combined'}.get(condition, condition.capitalize())
        title_parts = [condition_label]
        if rf_range is not None:
            title_parts.append(f'RF in [{rf_range[0]}, {rf_range[1]}]')
        fig.suptitle(f'Preferred Distance ({" | ".join(title_parts)})',
                     fontsize=18, color='black', y=1.02, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=4)

        plt.tight_layout()

        return fig, ax

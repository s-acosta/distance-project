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
from scipy.stats import binned_statistic_2d
from src.viz.CellVisualizer import CellVisualizer
from matplotlib.patches import Polygon

class SessionVisualizer():
    """
    Unified visualizer for single-session and multi-session DataFrames.

    Automatically detects mode based on DataFrame contents:
    - Single-session: Uses Suite2P coordinates on mean image background
    - Multi-session: Uses widefield coordinates on widefield image background

    Parameters
    ----------
    session_df : pd.DataFrame
        Session DataFrame with cell statistics. Must contain 'session_name' column.
        For multi-session, must also contain 'wf_coords' column.
    mouse_name : str, optional
        Required for multi-session mode. Mouse identifier (e.g., 'SAM041')
    experiment_name : str, optional
        Required for multi-session mode. Experiment name (e.g., 'tiling_120um')
    base_data_dir : str, optional
        Base directory for processed session data (Suite2P)
    base_multi_dir : str, optional
        Base directory for multi-session/widefield data

    Examples
    --------
    Single-session:
    >>> session_df = load_session_dataframe(...)  # has single session_name value
    >>> visualizer = SessionVisualizer(session_df)  # auto-detects single mode
    >>> visualizer.plot_reliable_cells_map()

    Multi-session:
    >>> combined_df = combiner.combine_sessions()  # has multiple session_name values
    >>> visualizer = SessionVisualizer(combined_df, mouse_name='SAM041', experiment_name='tiling_120um')
    >>> visualizer.plot_reliable_cells_map(show_areas=True)
    """

    def __init__(self, session_df, mouse_name=None, experiment_name=None,
                 base_data_dir=r'D:\data-distance-coding-project\processed-data',
                 base_multi_dir=r'D:\data-distance-coding-project\multi-session-data'):

        self.session_df = session_df
        self.mouse_name = mouse_name
        self.experiment_name = experiment_name
        self.base_data_dir = base_data_dir
        self.base_multi_dir = base_multi_dir

        # Detect mode based on session_name column
        self.mode = self._detect_mode()

        # Validate dependencies for detected mode
        self._validate_mode_requirements()

        # Load required data based on mode
        if self.mode == 'single':
            self.session_name = self.session_df['session_name'].iloc[0]
            self._suite2p_data = self._load_suite2p_data(self.session_name)
        else:
            self._widefield_data = self._load_widefield_data()

        self.distance_colors = setup_plot_style()
        self.modulation_colors = {
            'monocular': 'coral',
            'binocular': 'lightblue',
            'both': 'purple',
            'none': 'gray'
        }

    def _detect_mode(self):
        """
        Detect whether this is single-session or multi-session data.

        Returns
        -------
        str
            'single' or 'multi'
        """
        if 'session_name' not in self.session_df.columns:
            raise ValueError(
                "DataFrame must contain 'session_name' column. "
                "Add session_name to your DataFrame before creating SessionVisualizer."
            )

        n_sessions = self.session_df['session_name'].nunique()

        if n_sessions > 1:
            return 'multi'
        else:
            return 'single'

    def _validate_mode_requirements(self):
        """Validate that required dependencies are provided for the detected mode."""
        if self.mode == 'multi':
            if self.mouse_name is None or self.experiment_name is None:
                raise ValueError(
                    "Multi-session mode requires 'mouse_name' and 'experiment_name' parameters. "
                    f"Detected {self.session_df['session_name'].nunique()} unique sessions in DataFrame."
                )
            if 'wf_coords' not in self.session_df.columns:
                raise ValueError(
                    "Multi-session mode requires 'wf_coords' column in session_df. "
                    "Use SessionCombiner with widefield_mapper to add widefield coordinates."
                )

    def _load_suite2p_data(self, session_name):
        """
        Load Suite2P data for a single session.

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

    def _load_widefield_data(self):
        """
        Load widefield alignment data for multi-session mode.

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
        """Get the appropriate background image for the current mode."""
        if self.mode == 'single':
            return self._suite2p_data['ops']['refImg']
        else:
            return self._widefield_data['wf_maps']['img']

    def _get_background_extent(self):
        """Get the extent for imshow based on mode."""
        if self.mode == 'single':
            img = self._get_background_image()
            return [0, img.shape[1], img.shape[0], 0]
        else:
            return [0, 400, 400, 0]

    def _get_cell_coordinates(self, input_df):
        """
        Get (x, y) coordinates for cells based on current mode.

        Parameters
        ----------
        cell_ids : array-like
            List of cell IDs to get coordinates for

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (x_coords, y_coords) arrays
        """
        if self.mode == 'single':
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
        else:
            # Extract x, y from wf_coords column (each entry is [x, y])
            wf_coords = input_df['wf_coords'].values
            x_coords, y_coords = zip(*wf_coords)
            
            return x_coords, y_coords

    def _get_session_title(self):
        """Get appropriate title string for the current mode."""
        if self.mode == 'single':
            return self.session_name
        else:
            # Extract mouse name from session names (second part after splitting by underscore)
            session_names = self.session_df['session_name'].unique()
            mouse_names = list(set([s.split('_')[1] for s in session_names]))
            n_sessions = len(session_names)
            if len(mouse_names) == 1:
                return f"{mouse_names[0]} ({n_sessions} sessions)"
            else:
                return f"{', '.join(mouse_names)} ({n_sessions} sessions)"

    def _overlay_area_boundaries(self, ax, show_legend=True):
        """Overlay visual area boundaries on plot (multi-session only).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        show_legend : bool, optional
            Whether to show legend for area boundaries (default True).
        """
        if self.mode != 'multi':
            return

        areas_verts = self._widefield_data['areas_verts']

        # Define preferred order for areas
        area_order = ['V1', 'LM', 'RL', 'AM', 'PM', 'AL', 'None']

        # Sort areas by preferred order, putting unknown areas at the end
        sorted_areas = sorted(
            areas_verts.items(),
            key=lambda x: area_order.index(x[0]) if x[0] in area_order else len(area_order)
        )

        for i, (area_name, area_verts) in enumerate(sorted_areas):
            color = f"C{i%10}"
            patch = Polygon(area_verts, closed=True, facecolor='none',
                           edgecolor=color, linewidth=4, label=area_name, alpha=0.9)
            ax.add_patch(patch)

        if show_legend:
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='gray')
    
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
        Get list of all available visual area names in preferred order.

        Only applicable in multi-session mode.
        Areas are returned in order: V1, LM, RL, AM, PM, AL, None, then any others.

        Returns
        -------
        list
            List of area name strings.

        Raises
        ------
        ValueError
            If called in single-session mode.
        """
        if self.mode != 'multi':
            raise ValueError(
                "get_available_areas is only available in multi-session mode."
            )

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
   
    def filter_by_reliability(self, input_df, reliable_type='reliable',threshold=0.85):

        if reliable_type == 'reliable':
            output_df = input_df[
                (input_df['monocular'].apply(lambda x: x['reliable']) == True) &
                (input_df['binocular'].apply(lambda x: x['reliable']) == True) &
                ((input_df['monocular'].apply(lambda x: x['reliable_corr']) >= threshold) |
                (input_df['binocular'].apply(lambda x: x['reliable_corr']) >= threshold))
            ]

        elif reliable_type == 'not_reliable':
            output_df = input_df[
                (input_df['monocular'].apply(lambda x: x['reliable']) == False) &
                (input_df['binocular'].apply(lambda x: x['reliable']) == False) &
                (((input_df['monocular'].apply(lambda x: x['reliable_corr']) < threshold) &
                (input_df['binocular'].apply(lambda x: x['reliable_corr']) < threshold)))
            ]

        return output_df

    def filter_by_modulation(self, input_df, modulation_type='monocular'):
        
        reliable_df = self.filter_by_reliability(input_df, reliable_type='reliable',threshold=0.85)

        if modulation_type == 'monocular':
            output_df = reliable_df[
                (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
                (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05)
            ]

        elif modulation_type == 'binocular':
            output_df = reliable_df[
                (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05) &
                (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
            ]

        elif modulation_type == 'both':
            output_df = reliable_df[
                (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
                (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
            ]

        elif modulation_type == 'none':
            output_df = reliable_df[
                (reliable_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05) &
                (reliable_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05)
            ]

        return output_df

    def filter_by_area(self, input_df, area_name='V1'):

        output_df = input_df[input_df['area'] == area_name]

        return output_df

    def filter_by_session(self, input_df, session_name='SAM041_251128_001'):

        output_df = input_df[input_df['session_name'] == session_name]

        return output_df
    
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

    def plot_responsive_bars(self, group_by = None):

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

            n_reliable_cells = len(self.filter_by_reliability(group_df, reliable_type='reliable',threshold=0.85))
            total_cells = len(group_df)
            
            group_data[group] = {
                'total': total_cells,
                'reliable': n_reliable_cells
            }

        bar_width = 0.75
        for i, group in enumerate(groups):
            total = group_data[group]['total']
            reliable = group_data[group]['reliable']
            pct = reliable / total * 100 if total > 0 else 0

            # Plot total cells as grey bar
            ax.bar(group, total, color='lightgray', width=bar_width, label='Total Cells' if i == 0 else "")
            # Plot reliable cells as green bar
            ax.bar(group, reliable, color='lightgreen', width=bar_width, label='Reliable Cells' if i == 0 else "")

            # Add percentage text label
            ax.text(group, reliable, f'{pct:.1f}%\n({reliable}/{total})',
                   ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Number of Cells', fontsize=12)

        if group_by is 'session':
            # Split session names by underscore and show only 2nd and 3rd components
            groups = ['\n'.join(group.split('_')[3]) for group in groups]

        ax.set_xlabel(group_by.capitalize() if group_by is not None else 'All Cells', fontsize=12)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend()

        # add title with less padding so they appear closer
        fig.suptitle('Responsive Cells', fontsize=18, color='black', y=0.96)
        ax.set_title(self._get_session_title(), fontsize=14, color='grey', pad=2)

        plt.tight_layout()

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

    def plot_distance_bars(self, group_by=None, condition='both'):

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

            # Filter by modulation and get preferred distances
            modulated_df = self.filter_by_modulation(group_df, modulation_type=condition)
            modulated_count = len(modulated_df)

            if modulated_count > 0:
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
                'modulated': modulated_count,
                'distances': distance_counts
            }

        # Filter out groups with less than 10% reliable cells or less than 5 modulated cells
        groups = [g for g in groups if group_data[g]['total'] > 0 and
                  group_data[g]['reliable'] / group_data[g]['total'] >= 0.10 and
                  group_data[g]['modulated'] >= 5]

        # Set up bar positions with separation between groups
        x = np.arange(len(groups)) * 1.2
        bar_width = 0.20
        bar_gap = 0.03

        # Plot bars for each distance
        for j, dist in enumerate(distances):
            offset = (j - n_distances/2 + 0.5) * (bar_width + bar_gap)

            for i, group in enumerate(groups):
                modulated = group_data[group]['modulated']
                dist_count = group_data[group]['distances'][dist]
                pct = dist_count / modulated * 100 if modulated > 0 else 0

                # Plot distance cells as colored bar (percentage) with outline
                ax.bar(x[i] + offset, pct, color=self.distance_colors[j], width=bar_width,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=f'Distance {dist}' if i == 0 else "")

                # Add n cells as text label
                ax.text(x[i] + offset, pct + 1, f'n={dist_count}',
                       ha='center', va='bottom', fontsize=12, fontweight='medium')

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

        # ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray', fontsize=12)

        # add title with less padding so they appear closer
        condition_label = condition.capitalize() if condition != 'both' else 'Combined'
        fig.suptitle(f'Preferred Distance ({condition_label})', fontsize=22, color='black', y=0.96, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=20, color='grey', pad=2)

        plt.tight_layout()

        return fig

    def plot_mean_traces(self, condition='both', ax=None, device=None):
        """
        Plot mean traces for modulated cells.

        Note: This method requires raw trace data which is only available
        in single-session mode with a device provided.

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
        NotImplementedError
            If called in multi-session mode without device
        """
        if self.mode == 'multi':
            raise NotImplementedError(
                "plot_mean_traces requires raw trace data which is not available "
                "in multi-session mode. Filter to a single session and provide a device."
            )

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

    def plot_preferred_distance_polar(self, condition='binocular', ax = None):

        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1, polar=True)
        else:
            fig = ax.get_figure()

        ax.set_theta_zero_location('N')
        # Add gray background for central region (-25 to 25 degrees)
        ax.axvspan(np.radians(-25), np.radians(25), alpha=0.5, color='gray', zorder=0)
        # Add colored y-spans for distance ranges
        distance_colors = self.colors
        for dist in range(1, 5):
            ax.axhspan(dist - 0.30, dist + 0.30, alpha=0.4, color=distance_colors[dist-1], zorder=0)

        if condition is None:
            cells_to_plot = self.responsive_cells
        else:
            cells_to_plot = self.modulated_cells[condition]

        # Plot preferred distance and angle for each condition
        pref_distances = []
        pref_angles = []
        for cell in cells_to_plot:
            if condition is None:
                pref_distance = self.pullout_stat(cell_id=cell, condition='binocular', stat_type='pref_distance')
                pref_angle = self.pullout_stat(cell_id=cell, condition='binocular', stat_type='pref_angle')
            else:
                pref_distance = self.pullout_stat(cell_id=cell, condition=condition, stat_type='pref_distance')
                pref_angle = self.pullout_stat(cell_id=cell, condition=condition, stat_type='pref_angle')

            pref_distances.append(pref_distance)
            pref_angles.append(pref_angle)

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

        # Show area boundaries for multi-session mode
        if show_areas and self.mode == 'multi':
            self._overlay_area_boundaries(ax)

        # Axis adjustments differ by mode
        if self.mode == 'single':
            ax.invert_xaxis()
            ax.invert_yaxis()

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
            origin='upper' if self.mode == 'multi' else 'lower',
            extent=[0, ref_img.shape[1], 0, ref_img.shape[0]] if self.mode == 'single' else extent
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

        # Show area boundaries for multi-session mode
        if show_areas and self.mode == 'multi':
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
            for i, cell_id in enumerate(none_cells):
                ax.scatter(
                    none_x[i], none_y[i],
                    c='lightgray',
                    s=120, alpha=0.5,
                    edgecolor='black', linewidth=1
                )

        # Get coordinates for modulated cells and plot colored by distance
        if len(modulated_cells) > 0:
            mod_x, mod_y = self._get_cell_coordinates(modulated_cells)
            for i, cell_id in enumerate(modulated_cells):
                distance = self.extract_condition_feature(cell=[cell_id], condition=condition, feature='distance')
                ax.scatter(
                    mod_x[i], mod_y[i],
                    c=distance,
                    cmap='viridis',
                    vmin=0,
                    vmax=3,
                    s=120, alpha=1,
                    edgecolor='black', linewidth=1
                )

        # Show area boundaries for multi-session mode
        if show_areas and self.mode == 'multi':
            self._overlay_area_boundaries(ax)

        # Axis adjustments differ by mode
        if self.mode == 'single':
            ax.invert_xaxis()
            ax.invert_yaxis()

        # Remove axes and spines
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'{condition.capitalize()} Modulated Cells', fontsize=16)

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax

    def plot_cell_features(self, condition='binocular', feature='distance',
                           ax=None, show_areas=True, cmap='viridis'):
        """
        Plot cells on widefield map color-coded by a feature.

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
        else:
            # Filter by modulation type
            cells_df = self.filter_by_modulation(reliable_df, condition)

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
            cmap='viridis_r',
            vmin=vmin, vmax=vmax,
            s=50, alpha=0.85,
            edgecolor='black', linewidth=1.2
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(feature.capitalize(), fontsize=14)

        # Show area boundaries
        if show_areas and self.mode == 'multi':
            self._overlay_area_boundaries(ax)

        # Axis adjustments
        if self.mode == 'single':
            ax.invert_xaxis()
            ax.invert_yaxis()

        # Title
        if condition == 'reliable':
            condition_label = 'Reliable'
        elif condition == 'both':
            condition_label = 'Both'
        else:
            condition_label = condition.capitalize()
        fig.suptitle(f'{condition_label} Cells - {feature.capitalize()}', fontsize=22, color='black', y=0.96, fontweight='bold')
        ax.set_title(self._get_session_title(), fontsize=20, color='grey', pad=2)

        plt.tight_layout()

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

        # Show area boundaries for multi-session mode
        if show_areas and self.mode == 'multi':
            self._overlay_area_boundaries(ax)

        # Axis adjustments differ by mode
        if self.mode == 'single':
            ax.invert_xaxis()
            ax.invert_yaxis()

        ax.axis('off')
        plt.tight_layout(pad=0)

        return fig

    def plot_session_summary(self):

        fig_responsive_map, _ = self.plot_modulated_cells_map(condition=None)
        fig_binocular_map, _ = self.plot_modulated_cells_map(condition='binocular')
        fig_monocular_map, _ = self.plot_modulated_cells_map(condition='monocular')

        fig_distance_bars = plt.figure(figsize=(12, 4))
        gs = fig_distance_bars.add_gridspec(1, 3, hspace=0.3, wspace=0.5)
        # Create bar plot axes in second row
        axes_bar = []
        for i in range(3):
            ax = fig_distance_bars.add_subplot(gs[0, i])
            axes_bar.append(ax)

        # Call each plotting function with the appropriate axis
        self.plot_preferred_distance_bars(condition=None, ax=axes_bar[0])
        self.plot_preferred_distance_bars(condition='binocular', ax=axes_bar[1])
        self.plot_preferred_distance_bars(condition='monocular', ax=axes_bar[2])

        fig_distance_polar =plt.figure(figsize=(12, 18))
        gs = fig_distance_polar.add_gridspec(1, 3, hspace=0.3, wspace=0.5)

        # Create polar plots in first row
        axes_polar = []
        for i in range(3):
            ax = fig_distance_polar.add_subplot(gs[0, i], projection='polar')
            axes_polar.append(ax)

        self.plot_preferred_distance_polar(condition=None, ax=axes_polar[0])
        self.plot_preferred_distance_polar(condition='binocular', ax=axes_polar[1])
        self.plot_preferred_distance_polar(condition='monocular', ax=axes_polar[2])

        return fig_responsive_map, fig_binocular_map, fig_monocular_map, fig_distance_bars, fig_distance_polar

    def plot_modulation_fraction_by_angle(self, modulation_type='binocular',
                                           angle_range=(-30, 75), window_size=5,
                                           area=None, ax=None, show_density=True, color=None):
        """
        Plot fraction of modulated cells among reliable cells as a function of RF angle.

        Generalized version that works for any modulation type.

        Parameters
        ----------
        modulation_type : str
            Type of modulation ('binocular', 'monocular', 'both')
        angle_range : tuple, optional
            (min_angle, max_angle) range to plot (default: (-30, 75))
        window_size : float, optional
            Half-width of the sliding window in degrees (default: 5)
        area : str, optional
            Visual area to filter by (e.g., 'V1', 'LM'). If None, uses all areas.
            Only applicable in multi-session mode.
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
        ax.plot(angles, frac_modulated, color=color, lw=2, label=f'{modulation_type.capitalize()} fraction')
        ax.set_xlabel('RF Angle (degrees)', fontsize=14)
        ax.set_ylabel(f'Fraction {modulation_type.capitalize()} among Reliable', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
        ax.grid(alpha=0.3)
        #set lims from 0 to 0.5
        ax.set_ylim(0, 0.4)

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

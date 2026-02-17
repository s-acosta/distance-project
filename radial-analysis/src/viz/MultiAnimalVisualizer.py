import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt
from src.utils import setup_plot_style


class MultiAnimalVisualizer():
    """
    Visualizer for comparing results across multiple animals/recordings.

    Takes multiple combined DataFrames and plots comparative bar charts
    for a given brain area, with one group of bars per recording.

    Parameters
    ----------
    combined_dfs : list of pd.DataFrame
        List of combined DataFrames, one per recording.
    mouse_names : list of str
        List of mouse identifiers.
    experiment_names : list of str
        List of experiment identifiers.
    """

    def __init__(self, combined_dfs, mouse_names, experiment_names):

        self.combined_dfs = combined_dfs
        self.mouse_names = mouse_names
        self.experiment_names = experiment_names
        self.n_recordings = len(combined_dfs)

        self.distance_colors = setup_plot_style()
        self.modulation_colors = {
            'monocular': 'coral',
            'binocular': 'lightblue',
            'both': 'purple',
            'none': 'gray'
        }

        # Area colors matching widefield border colors (same order as SessionVisualizer)
        area_order = ['V1', 'LM', 'RL', 'AM', 'PM', 'AL', 'None']
        self.area_colors = {area: f"C{i%10}" for i, area in enumerate(area_order)}

    # ---- Filtering helpers (standalone, no file I/O) ----

    def filter_by_area(self, input_df, area_name='V1'):
        return input_df[input_df['area'] == area_name]

    def filter_by_reliability(self, input_df, reliable_type='reliable', threshold=0.85):
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
        reliable_df = self.filter_by_reliability(input_df, reliable_type='reliable', threshold=0.85)

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

    def _get_label(self, idx):
        return self.mouse_names[idx]

    # ---- Plotting methods ----

    def plot_responsive_bars(self, area):

        fig, ax = plt.subplots(figsize=(max(5, self.n_recordings * 2), 5))

        bar_width = 0.75
        labels = []

        for i in range(self.n_recordings):
            area_df = self.filter_by_area(self.combined_dfs[i], area_name=area)
            reliable_df = self.filter_by_reliability(area_df, reliable_type='reliable', threshold=0.85)

            total = len(area_df)
            reliable = len(reliable_df)
            pct = reliable / total * 100 if total > 0 else 0

            label = self._get_label(i)
            labels.append(label)

            bar_color = self.area_colors.get(area, 'lightgreen')

            ax.bar(i, total, color='lightgray', width=bar_width, label='Total Cells' if i == 0 else "")
            ax.bar(i, reliable, color=bar_color, width=bar_width, alpha=0.7, label='Reliable Cells' if i == 0 else "")

            ax.text(i, reliable / 2, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=16, fontweight='bold')
            ax.text(i, reliable + 1, f'({reliable}/{total})',
                    ha='center', va='bottom', fontsize=16)

        ax.set_xticks(range(self.n_recordings))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax.set_ylabel('Number of Cells', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)

        fig.suptitle(f'Responsive Cells - {area}', fontsize=18, color='black', y=0.96)
        plt.tight_layout()

        return fig

    def plot_modulation_bars(self, area):

        modulation_types = ['monocular', 'binocular', 'both']
        n_modulations = len(modulation_types)

        fig, ax = plt.subplots(figsize=(max(8, self.n_recordings * 2.5), 7))

        x = np.arange(self.n_recordings) * 1.2
        bar_width = 0.28
        bar_gap = 0.03
        labels = []

        for i in range(self.n_recordings):
            labels.append(self._get_label(i))

        for j, mod_type in enumerate(modulation_types):
            offset = (j - n_modulations / 2 + 0.5) * (bar_width + bar_gap)

            for i in range(self.n_recordings):
                area_df = self.filter_by_area(self.combined_dfs[i], area_name=area)
                reliable_df = self.filter_by_reliability(area_df, reliable_type='reliable', threshold=0.85)
                modulated_df = self.filter_by_modulation(area_df, modulation_type=mod_type)

                reliable_count = len(reliable_df)
                mod_count = len(modulated_df)
                pct = mod_count / reliable_count * 100 if reliable_count > 0 else 0

                ax.bar(x[i] + offset, mod_count, color=self.modulation_colors[mod_type], width=bar_width,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=mod_type.capitalize() if i == 0 else "")

                ax.text(x[i] + offset, mod_count / 2, f'{pct:.1f}%',
                        ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)

        ax.set_ylabel('Number of Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=16)

        plt.tight_layout()
        fig.suptitle(f'Modulated Cells - {area}', fontsize=22, color='black', y=1.1, fontweight='bold')

        return fig

    def plot_distance_bars(self, area, condition='both'):

        distances = [0, 1, 2, 3]
        n_distances = len(distances)

        fig, ax = plt.subplots(figsize=(max(8, self.n_recordings * 2.5), 7))

        x = np.arange(self.n_recordings) * 1.2
        bar_width = 0.20
        bar_gap = 0.03
        labels = []

        for i in range(self.n_recordings):
            labels.append(self._get_label(i))

        for j, dist in enumerate(distances):
            offset = (j - n_distances / 2 + 0.5) * (bar_width + bar_gap)

            for i in range(self.n_recordings):
                area_df = self.filter_by_area(self.combined_dfs[i], area_name=area)
                modulated_df = self.filter_by_modulation(area_df, modulation_type=condition)
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
                    dist_count = np.sum(np.array(pref_distances) == dist)
                else:
                    dist_count = 0

                pct = dist_count / modulated_count * 100 if modulated_count > 0 else 0

                ax.bar(x[i] + offset, dist_count, color=self.distance_colors[j], width=bar_width,
                       edgecolor='black', linewidth=1.2, alpha=0.85,
                       label=f'Distance {dist}' if i == 0 else "")

                ax.text(x[i] + offset, dist_count / 2, f'{pct:.1f}%',
                        ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)

        ax.set_ylabel('Number of Cells', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        condition_label = condition.capitalize() if condition != 'both' else 'Combined'
        fig.suptitle(f'Preferred Distance ({condition_label}) - {area}', fontsize=22, color='black', y=0.96, fontweight='bold')

        plt.tight_layout()

        return fig

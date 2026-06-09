import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

class WidefieldVisualizer:
    """
    Visualizes widefield alignment results, visual areas, and cell locations.

    This class provides visualization tools for:
    - Widefield maps with visual area boundaries
    - Aligned sessions overlaid on widefield
    - Individual cells from dataframes plotted in widefield space

    Parameters
    ----------
    animal_id : str
        Animal identifier (e.g., 'SAM041')
    experiment_name : str
        Experiment name (e.g., 'tiling_120um')

    Attributes
    ----------
    aligner_params : dict
        Dictionary containing widefield maps, area vertices, and transformation parameters
    base_multi_dir : str
        Base directory for multi-session data
    base_processed_dir : str
        Base directory for processed session data

    Examples
    --------
    >>> visualizer = WidefieldVisualizer('SAM041', 'tiling_120um')
    >>> visualizer.plot_widefield_with_areas()
    >>> visualizer.plot_cells_in_widefield(session_df, session_name='2024-01-15_SAM041_radial')
    """

    def __init__(self, animal_id, experiment_name):

        self.animal_id = animal_id
        self.experiment_name = experiment_name

        self.base_multi_dir = r'D:/data-distance-coding-project/multi-session-data/'
        self.base_processed_dir = r'D:/data-distance-coding-project/processed-data/'
        self.multi_session_dir = os.path.join(self.base_multi_dir, animal_id + '_' + experiment_name)

        # Load alignment parameters
        self.aligner_params = self._load_alignment_parameters()

    def _load_alignment_parameters(self):
        """Load alignment parameters from saved pickle file."""

        params_path = os.path.join(
            self.multi_session_dir,
            f'{self.animal_id}_{self.experiment_name}_alignment_parameters.pkl'
        )

        try:
            with open(params_path, 'rb') as f:
                aligner_params = pickle.load(f)
            print(f"Alignment parameters loaded from: {params_path}")
            return aligner_params
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Alignment parameters not found at {params_path}. "
                "Please run WidefieldAligner first to create alignment."
            )

    def plot_widefield_with_areas(self, map_type='img', save_path=None):
        """
        Plot widefield image with visual area boundaries.

        Parameters
        ----------
        map_type : str, optional
            Which map to display as background. Options: 'img' (default), 'azi', 'alt', 'VFS'.
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

        # Get map image
        widefield_img = self.aligner_params['wf_maps'][map_type]
        cmap = 'gray' if map_type == 'img' else 'jet'
        im = ax.imshow(widefield_img, cmap=cmap, alpha=1,
                 extent=[0, 400, 400, 0], origin='upper')

        # Add colorbar for retinotopic maps
        if map_type != 'img':
            plt.colorbar(im, ax=ax, shrink=0.7)

        # Overlay area boundaries
        areas_verts = self.aligner_params['areas_verts']
        use_colored = (map_type == 'img')
        for i, (area_name, area_verts) in enumerate(areas_verts.items()):
            if use_colored:
                color = f"C{i%10}"
                patch = Polygon(area_verts, closed=True, facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=3, label=area_name)
            else:
                patch = Polygon(area_verts, closed=True, facecolor='none',
                              edgecolor='black', linewidth=4)
            ax.add_patch(patch)

        # Add legend only for default img mode
        if use_colored:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                     loc='upper right', fontsize=12)

        ax.set_title(f'Widefield Map with Visual Areas\n{self.animal_id} - {self.experiment_name}',
                    fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return fig, ax

    def plot_sessions_on_widefield(self, session_names=None, session_alpha=0.5, save_path=None):
        """
        Plot aligned sessions overlaid on widefield image.

        Parameters
        ----------
        session_names : list of str, optional
            List of session names to plot. If None, plots all sessions.
        session_alpha : float, optional
            Transparency for session overlays (0.0 to 1.0). Default is 0.6.
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """

        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)

            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy

            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed

        # Get widefield reference
        widefield = self.aligner_params['wf_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())

        # Create composite image starting with widefield in grayscale
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

        # Determine which sessions to plot
        wf_transformations = self.aligner_params['wf_transformations']
        if session_names is None:
            session_names = list(wf_transformations.keys())

        # Overlay sessions in green
        for session_name in session_names:
            if session_name not in wf_transformations:
                print(f"Warning: {session_name} not found in alignment parameters. Skipping.")
                continue

            params = wf_transformations[session_name]

            # Load session mean image
            session_dir = os.path.join(self.base_processed_dir, session_name)
            ops_path = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')

            try:
                ops = np.load(ops_path, allow_pickle=True).item()
                mean_img = ops['meanImg']
            except FileNotFoundError:
                print(f"Warning: Could not load mean image for {session_name}. Skipping.")
                continue

            # Normalize and resize
            mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
            mean_img_resized = cv2.resize(mean_img_norm,
                                        (widefield.shape[1], widefield.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

            # Transform
            transformed = transform_image(
                mean_img_resized,
                params['scale'],
                params['rotation'],
                params['dx'],
                params['dy']
            )

            # Blend using alpha compositing in green
            session_green = np.zeros_like(composite)
            session_green[:, :, 1] = transformed
            mask = (transformed > 0).astype(float)
            composite = composite * (1 - session_alpha * mask[:, :, None]) + session_green * session_alpha

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        ax.imshow(composite, extent=[0, 400, 400, 0], origin='upper')

        ax.set_title(f'Aligned Sessions on Widefield\n{self.animal_id} - {self.experiment_name}',
                    fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.axis('off')

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return fig, ax

    def _draw_widefield_to_ax(self, ax, session_names=None, session_alpha=0.6):
        """Draw widefield composite with area boundaries and sessions onto an existing axis."""

        def transform_image(img, scale, rotation, dx, dy):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            return cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        widefield = self.aligner_params['wf_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

        wf_transformations = self.aligner_params['wf_transformations']
        if session_names is None:
            session_names = list(wf_transformations.keys())

        for session_name in session_names:
            if session_name not in wf_transformations:
                print(f"Warning: {session_name} not found in alignment parameters. Skipping.")
                continue
            params = wf_transformations[session_name]
            session_dir = os.path.join(self.base_processed_dir, session_name)
            ops_path = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')
            try:
                ops = np.load(ops_path, allow_pickle=True).item()
                mean_img = ops['meanImg']
            except FileNotFoundError:
                print(f"Warning: Could not load mean image for {session_name}. Skipping.")
                continue
            mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
            mean_img_resized = cv2.resize(mean_img_norm, (widefield.shape[1], widefield.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            transformed = transform_image(mean_img_resized, params['scale'], params['rotation'],
                                          params['dx'], params['dy'])
            session_green = np.zeros_like(composite)
            session_green[:, :, 1] = transformed
            mask = (transformed > 0).astype(float)
            composite = composite * (1 - session_alpha * mask[:, :, None]) + session_green * session_alpha

        ax.imshow(composite, extent=[0, 400, 400, 0], origin='upper')

        areas_verts = self.aligner_params['areas_verts']
        for i, (area_name, area_verts) in enumerate(areas_verts.items()):
            patch = Polygon(area_verts, closed=True, facecolor='none',
                            edgecolor='white', linewidth=5, label=area_name)
            ax.add_patch(patch)

        ax.set_title(f'{self.animal_id}', fontsize=13)
        ax.axis('off')

    def plot_widefield_with_areas_and_sessions(self, session_names=None, session_alpha=0.6, save_path=None):
        """
        Plot widefield with both visual area boundaries and aligned sessions.

        Parameters
        ----------
        session_names : list of str, optional
            List of session names to plot. If None, plots all sessions.
        session_alpha : float, optional
            Transparency for session overlays (0.0 to 1.0). Default is 0.6.
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        self._draw_widefield_to_ax(ax, session_names=session_names, session_alpha=session_alpha)
        ax.set_title(f'Widefield Map with Areas and Aligned Sessions\n{self.animal_id} - {self.experiment_name}',
                     fontsize=14)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()
        return fig, ax

    def plot_cells_in_widefield(self, session_df, session_name=None,
                                show_areas=True, show_session=True,
                                cell_color='red', cell_size=50,
                                save_path=None):
        """
        Plot cells from a dataframe on the widefield map.

        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame with cell data. Must contain 'wf_coords' column with
            widefield coordinates [xwf, ywf] for each cell.
        session_name : str, optional
            Session name to overlay the session image. If None, only shows widefield.
        show_areas : bool, optional
            Whether to show visual area boundaries. Default is True.
        show_session : bool, optional
            Whether to overlay the session mean image. Default is True.
            Requires session_name to be provided.
        cell_color : str, optional
            Color for cell markers. Default is 'red'.
        cell_size : int, optional
            Size of cell markers. Default is 50.
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """

        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)

            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy

            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed

        # Get widefield reference
        widefield = self.aligner_params['wf_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())

        # Create composite image
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

        # Overlay session if requested
        if show_session and session_name is not None:
            wf_transformations = self.aligner_params['wf_transformations']

            if session_name not in wf_transformations:
                print(f"Warning: {session_name} not found in alignment parameters. Skipping session overlay.")
            else:
                params = wf_transformations[session_name]

                # Load session mean image
                session_dir = os.path.join(self.base_processed_dir, session_name)
                ops_path = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')

                try:
                    ops = np.load(ops_path, allow_pickle=True).item()
                    mean_img = ops['meanImg']

                    # Normalize and resize
                    mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
                    mean_img_resized = cv2.resize(mean_img_norm,
                                                (widefield.shape[1], widefield.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)

                    # Transform
                    transformed = transform_image(
                        mean_img_resized,
                        params['scale'],
                        params['rotation'],
                        params['dx'],
                        params['dy']
                    )

                    # Blend in green
                    session_alpha = 0.5
                    session_green = np.zeros_like(composite)
                    session_green[:, :, 1] = transformed
                    mask = (transformed > 0).astype(float)
                    composite = composite * (1 - session_alpha * mask[:, :, None]) + session_green * session_alpha

                except FileNotFoundError:
                    print(f"Warning: Could not load mean image for {session_name}. Skipping session overlay.")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        ax.imshow(composite, extent=[0, 400, 400, 0], origin='upper')

        # Overlay area boundaries if requested
        if show_areas:
            areas_verts = self.aligner_params['areas_verts']
            for i, (area_name, area_verts) in enumerate(areas_verts.items()):
                color = f"C{i%10}"
                patch = Polygon(area_verts, closed=True, facecolor='none',
                              edgecolor=color, linewidth=2, label=area_name, alpha=0.7)
                ax.add_patch(patch)

        # Plot cells
        if 'wf_coords' not in session_df.columns:
            raise ValueError(
                "DataFrame must contain 'wf_coords' column. "
                "Use WidefieldTransformer.append_widefield_to_session_df() first."
            )

        wf_coords = np.array(session_df['wf_coords'].tolist())
        xwf = wf_coords[:, 0]
        ywf = wf_coords[:, 1]

        ax.scatter(xwf, ywf, c=cell_color, s=cell_size,
                  alpha=0.7, edgecolors='white', linewidths=1,
                  label=f'Cells (n={len(session_df)})', zorder=10)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        title = f'Cells in Widefield Space\n{self.animal_id} - {self.experiment_name}'
        if session_name is not None:
            title += f'\nSession: {session_name}'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        # ax.set_xlim(0, 400)
        # ax.set_ylim(0, 400)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return fig, ax

    def plot_cells_by_area(self, session_df, session_name=None,
                          show_session=True, save_path=None):
        """
        Plot cells colored by their visual area assignment.

        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame with cell data. Must contain 'wf_coords' and 'area' columns.
        session_name : str, optional
            Session name to overlay the session image. If None, only shows widefield.
        show_session : bool, optional
            Whether to overlay the session mean image. Default is True.
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """

        if 'area' not in session_df.columns:
            raise ValueError(
                "DataFrame must contain 'area' column. "
                "Use WidefieldTransformer to assign areas first."
            )

        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)

            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy

            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed

        # Get widefield reference
        widefield = self.aligner_params['wf_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())

        # Create composite image
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

        # Overlay session if requested
        if show_session and session_name is not None:
            wf_transformations = self.aligner_params['wf_transformations']

            if session_name in wf_transformations:
                params = wf_transformations[session_name]

                session_dir = os.path.join(self.base_processed_dir, session_name)
                ops_path = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')

                try:
                    ops = np.load(ops_path, allow_pickle=True).item()
                    mean_img = ops['meanImg']

                    mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
                    mean_img_resized = cv2.resize(mean_img_norm,
                                                (widefield.shape[1], widefield.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)

                    transformed = transform_image(
                        mean_img_resized,
                        params['scale'],
                        params['rotation'],
                        params['dx'],
                        params['dy']
                    )

                    session_alpha = 0.4
                    session_green = np.zeros_like(composite)
                    session_green[:, :, 1] = transformed
                    mask = (transformed > 0).astype(float)
                    composite = composite * (1 - session_alpha * mask[:, :, None]) + session_green * session_alpha

                except FileNotFoundError:
                    pass

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        ax.imshow(composite, extent=[0, 400, 400, 0], origin='upper')

        # Overlay area boundaries
        areas_verts = self.aligner_params['areas_verts']
        area_colors = {}
        for i, (area_name, area_verts) in enumerate(areas_verts.items()):
            color = f"C{i%10}"
            area_colors[area_name] = color
            patch = Polygon(area_verts, closed=True, facecolor='none',
                          edgecolor=color, linewidth=4, alpha=0.7)
            ax.add_patch(patch)

        # Plot cells colored by area
        wf_coords = np.array(session_df['wf_coords'].tolist())

        # Plot unassigned cells in grey (lower zorder so they sit behind assigned cells)
        unassigned_mask = session_df['area'].isna() | (session_df['area'] == 'None') | (~session_df['area'].isin(area_colors.keys()))
        unassigned_coords = wf_coords[unassigned_mask]
        if len(unassigned_coords) > 0:
            ax.scatter(unassigned_coords[:, 0], unassigned_coords[:, 1],
                      c='grey', s=50,
                      alpha=0.6, edgecolors='white', linewidths=1,
                      label=f'Unassigned (n={len(unassigned_coords)})', zorder=9)

        for area_name in session_df['area'].unique():
            if pd.isna(area_name) or area_name == 'None' or area_name not in area_colors:
                continue

            mask = session_df['area'] == area_name
            area_coords = wf_coords[mask]

            if len(area_coords) > 0:
                ax.scatter(area_coords[:, 0], area_coords[:, 1],
                          c=area_colors[area_name], s=50,
                          alpha=0.8, edgecolors='white', linewidths=1,
                          label=f'{area_name} (n={len(area_coords)})', zorder=10)

        # Add legend (only cell scatter entries, on top of everything)
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper right', fontsize=14)
        legend.set_zorder(20)

        title = f'Cells by Visual Area\n{self.animal_id} - {self.experiment_name}'
        if session_name is not None:
            title += f'\nSession: {session_name}'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return fig, ax

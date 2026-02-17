import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Polygon
from matplotlib.widgets import LassoSelector, Slider
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path

class WidefieldAligner:
    """
    Aligns multiple two-photon imaging sessions to widefield retinotopy maps.

    This class provides interactive tools for:
    - Loading and visualizing widefield maps
    - Drawing visual area boundaries (V1, LM, AL, RL, AM, PM)
    - Aligning session mean images to widefield reference using scale, rotation, and translation
    - Exporting alignment parameters for downstream analysis

    Parameters
    ----------
    animal_id : str
        Animal identifier (e.g., 'SAM041')
    experiment_name : str
        Experiment name (e.g., 'tiling_120um')

    Attributes
    ----------
    base_multi_dir : str
        Base directory for multi-session data
    base_processed_dir : str
        Base directory for processed session data
    session_names : list
        List of session names to align
    mean_images : list
        Suite2p mean images for each session
    widefield_maps : dict
        Dictionary containing widefield image and retinotopy maps
    areas_verts : dict
        Dictionary mapping area names to polygon vertices
    alignment_params : dict
        Dictionary of alignment parameters for each session

    Examples
    --------
    >>> aligner = WidefieldAligner('SAM041', 'tiling_120um')
    >>> # Interactive alignment is performed during initialization
    >>> # Export results
    >>> aligner.export_alignment_parameters()
    >>> # Plot final result
    >>> aligner.plot_widefield_with_areas()
    """

    def __init__(self, animal_id, experiment_name):

        self.base_multi_dir = r'D:/data-distance-coding-project/multi-session-data/'
        self.base_processed_dir = r'D:/data-distance-coding-project/processed-data/'
        self.multi_session_dir = os.path.join(self.base_multi_dir, animal_id + '_' + experiment_name)

        self.animal_id = animal_id
        self.experiment_name = experiment_name

        self.session_names = self.read_config_file()
        self.mean_images = self.load_mean_images()
        self.widefield_maps = self.load_widefield_maps()

        try:
            self.aligner_params = self.import_alignment_parameters()
            print(f"Alignment parameters loaded from: {self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl'}")
        except:
            print("No alignment parameters found. Drawing areas and aligning sessions.")
            self.areas_verts = self.draw_areas()
            self.wf_transformations = self.align_all_sessions_to_widefield()
            self.export_alignment_parameters()
            self.aligner_params = self.import_alignment_parameters()

    def read_config_file(self, multi_session_info_file = None):

        if multi_session_info_file is None:
            session_info_file = os.path.join(self.multi_session_dir, self.animal_id + '_' + self.experiment_name + '_sessions.txt')
        else:
            session_info_file = multi_session_info_file

        session_names = []
        try:
            with open(session_info_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Multi-session session info file not found: {session_info_file}")
            raise
        except Exception as e:
            print(f"Error reading multi-session session info file {session_info_file}: {e}")
            raise

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                session_name_in_commas = line.split(',')
                mouse_name, record_date, experiment_name = session_name_in_commas
                session_name = record_date + '_' + mouse_name + '_' + experiment_name
                session_names.append(session_name)

        return session_names

    def load_mean_images(self):

        self.mean_images = []
        for session_name in self.session_names:
            session_dir = self.get_session_dir(session_name)
            suite2p_dir = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')
            if not os.path.exists(suite2p_dir):
                raise FileNotFoundError(f"Suite2p directory does not exist: {suite2p_dir}")
            ops = np.load(suite2p_dir, allow_pickle=True).item()
            self.mean_images.append(ops['meanImg'])

        return self.mean_images

    def get_session_dir(self, session_name):

        session_dir = os.path.join(self.base_processed_dir, session_name)

        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

        return session_dir

    def load_widefield_maps(self):

        widefield_maps = {}

        # Load the snap.mat file from the animal directory
        snap_mat_path = os.path.join(self.multi_session_dir, 'snap.mat')
        snap_data = loadmat(snap_mat_path)
        widefield_img = snap_data['snap']
        # resize widefield_img to the size of 400x400
        widefield_img = cv2.resize(widefield_img, (400, 400))
        widefield_maps['img'] = widefield_img

        maps_mat_path = os.path.join(self.multi_session_dir, 'maps.mat')
        maps_data = loadmat(maps_mat_path, struct_as_record=False, squeeze_me=True)
        VFS_raw = maps_data['maps'].VFS_raw
        VFS_filtered = gaussian_filter(VFS_raw, sigma=1.5)
        widefield_maps['VFS'] = VFS_filtered

        alt_raw = maps_data['maps'].VerticalRetinotopy
        alt_filtered = gaussian_filter(alt_raw, sigma=1.5)
        widefield_maps['alt'] = alt_filtered

        azi_raw = maps_data['maps'].HorizontalRetinotopy
        azi_filtered = gaussian_filter(azi_raw, sigma=1.5)
        widefield_maps['azi'] = azi_filtered

        return widefield_maps

    def draw_areas(self):

        area_names = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']
        VFS_map = self.widefield_maps['VFS']

        fig, window_ax = plt.subplots(figsize=(8, 8), dpi=100)
        window_ax.imshow(
            VFS_map,
            cmap='jet',
            extent=[0, 400, 400, 0],
            origin='upper'
        )
        # window_ax.set_xlim(0, 400)
        # window_ax.set_ylim(0, 400)
        window_ax.set_xlabel("X (units)")
        window_ax.set_ylabel("Y (units)")

        im = window_ax.imshow(
            VFS_map,
            cmap='jet',
            extent=[0, 400, 400, 0],
            origin='upper',
            alpha=0.4
        )

        # --- Improved lasso/area drawing logic ---
        current_area_idx = [0]
        areas_verts = {}
        patches_dict = {}

        def draw_all_patches():
            for i, (area_name, area_verts) in enumerate(areas_verts.items()):
                color = f"C{i%10}"
                patch = Polygon(area_verts, closed=True, facecolor='none', edgecolor=color, linewidth=3, label=area_name)
                window_ax.add_patch(patch)
                patches_dict[area_name] = patch
            # Remove duplicate legends
            handles, labels = window_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            window_ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            fig.canvas.draw_idle()

        def onselect(verts):
            area = area_names[current_area_idx[0]]
            areas_verts[area] = list(verts)
            draw_all_patches()
            # Move to next area if not done
            if current_area_idx[0] < len(area_names) - 1:
                current_area_idx[0] += 1
                print(f"Draw lasso for: {area_names[current_area_idx[0]]}")
                # Update the title to instruct the user
                window_ax.set_title(area_names[current_area_idx[0]])
            else:
                print("All areas drawn. Close the figure window to continue.")
                window_ax.set_title("All areas drawn. Close the figure window to continue.")
            fig.canvas.draw_idle()

        # Set initial title with instruction
        window_ax.set_title(
            f"All Images on Figure Canvas (units)\nDraw lasso for: {area_names[0]}"
        )

        print(f"Draw lasso for: {area_names[0]}")
        lasso = LassoSelector(window_ax, onselect, props=dict(color='black', linewidth=3))

        # If any areas already have verts (e.g., from a previous run), show them
        if areas_verts:
            draw_all_patches()

        plt.show()

        return areas_verts

    def align_session_to_widefield(self, session_id, slider_limit_px=500):
        """
        Align a single session's mean image to the widefield image with scale, rotation, and translation.
        Previously aligned sessions remain visible in green overlay.

        Parameters
        ----------
        session_id : int
            The session index
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.

        Returns
        -------
        dict
            Dictionary containing transformation parameters:
            {'scale': float, 'rotation': float, 'dx': float, 'dy': float}
        """


        scale0 = 0.33
        rotation0 = 174

        dx0, dy0 = 0.0, 0.0
        alpha0 = 0.5

        # Helper function to apply transformation to an image
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)

            # Create transformation matrix
            # First scale and rotate around center
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)

            # Then add translation
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy

            # Apply transformation
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed

        # Get the widefield reference image
        widefield = self.widefield_maps['img'].copy()

        # Normalize widefield to 0-1 for display
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())

        # Get the mean image to align
        mean_img = self.mean_images[session_id].copy()

        # Normalize mean image to 0-1
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())

        # Resize mean_img to match widefield size
        mean_img_resized = cv2.resize(mean_img_norm,
                                    (widefield.shape[1], widefield.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

        # Create green colormap for mean images
        green_cmap = LinearSegmentedColormap.from_list('green', ['black', 'lime'])

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)

        # Create initial composite
        # Start with widefield in grayscale
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

        # Overlay previously aligned sessions in green (dimmer, underneath)
        if hasattr(self, 'field_coordinates_widefield'):
            for prev_session_id, params in self.field_coordinates_widefield.items():
                if prev_session_id != session_id:  # Don't show the current one yet
                    prev_img = self.mean_images[prev_session_id].copy()
                    prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                    prev_img_resized = cv2.resize(prev_img_norm,
                                                (widefield.shape[1], widefield.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                    prev_transformed = transform_image(
                        prev_img_resized,
                        params['scale'],
                        params['rotation'],
                        params['dx'],
                        params['dy']
                    )
                    # Blend green on top of grayscale using alpha compositing
                    prev_alpha = 0.3
                    prev_green = np.zeros_like(composite)
                    prev_green[:, :, 1] = prev_transformed
                    composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha

        # Add current session on top (brighter)
        current_transformed = transform_image(mean_img_resized, scale0, rotation0, dx0, dy0)
        current_green = np.zeros_like(composite)
        current_green[:, :, 1] = current_transformed
        current_mask = (current_transformed > 0).astype(float)
        composite = composite * (1 - alpha0 * current_mask[:, :, None]) + current_green * alpha0

        img_display = ax.imshow(composite)
        ax.set_title(f"Aligning session: {self.session_names[session_id]}")
        ax.axis('off')

        # Create sliders
        ax_scale = plt.axes([0.15, 0.25, 0.7, 0.02])
        ax_rotation = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_dx = plt.axes([0.15, 0.15, 0.7, 0.02])
        ax_dy = plt.axes([0.15, 0.10, 0.7, 0.02])
        ax_alpha = plt.axes([0.15, 0.05, 0.7, 0.02])

        slider_scale = Slider(ax_scale, 'Scale', 0.1, 1.0, valinit=scale0, valstep=0.01)
        slider_rotation = Slider(ax_rotation, 'Rotation (°)', -180, 180, valinit=rotation0, valstep=1)
        slider_dx = Slider(ax_dx, 'dx (px)', -slider_limit_px, slider_limit_px, valinit=dx0, valstep=1)
        slider_dy = Slider(ax_dy, 'dy (px)', -slider_limit_px, slider_limit_px, valinit=dy0, valstep=1)
        slider_alpha = Slider(ax_alpha, 'Transparency', 0.0, 1.0, valinit=alpha0, valstep=0.05)

        def update(val):
            """Update the display when sliders change."""
            scale = slider_scale.val
            rotation = slider_rotation.val
            dx = slider_dx.val
            dy = slider_dy.val
            alpha = slider_alpha.val

            # Recreate composite
            composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)

            # Overlay previously aligned sessions (dimmer, underneath)
            if hasattr(self, 'field_coordinates_widefield'):
                for prev_numeral, params in self.field_coordinates_widefield.items():
                    if prev_session_id != session_id:
                        prev_img = self.mean_images[prev_numeral].copy()
                        prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                        prev_img_resized = cv2.resize(prev_img_norm,
                                                    (widefield.shape[1], widefield.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                        prev_transformed = transform_image(
                            prev_img_resized,
                            params['scale'],
                            params['rotation'],
                            params['dx'],
                            params['dy']
                        )
                        # Add to green channel with reduced intensity
                        prev_alpha = 0.3
                        prev_green = np.zeros_like(composite)
                        prev_green[:, :, 1] = prev_transformed
                        composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha

            # Add current session on top (brighter)
            current_transformed = transform_image(mean_img_resized, scale, rotation, dx, dy)
            current_green = np.zeros_like(composite)
            current_green[:, :, 1] = current_transformed
            current_mask = (current_transformed > 0).astype(float)
            composite = composite * (1 - alpha * current_mask[:, :, None]) + current_green * alpha

            img_display.set_data(composite)
            ax.set_title(
                f"Aligning session: {self.session_names[session_id]} | "
                f"Scale: {scale:.2f}, Rotation: {rotation:.0f}°, "
                f"dx: {dx:.0f}px, dy: {dy:.0f}px"
            )
            fig.canvas.draw_idle()

        # Connect sliders to update function
        slider_scale.on_changed(update)
        slider_rotation.on_changed(update)
        slider_dx.on_changed(update)
        slider_dy.on_changed(update)
        slider_alpha.on_changed(update)

        plt.show(block=True)

        # Return the transformation parameters
        return {
            'scale': slider_scale.val,
            'rotation': slider_rotation.val,
            'dx': slider_dx.val,
            'dy': slider_dy.val
        }

    def align_all_sessions_to_widefield(self, slider_limit_px=500):
        """
        Sequentially align all sessions to the widefield image.
        Each aligned session remains visible while aligning subsequent ones.

        Parameters
        ----------
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.

        Returns
        -------
        dict
            Dictionary mapping session numeral to transformation parameters
        """
        # Initialize storage for transformations

        wf_transformations = {}

        print("Starting sequential alignment to widefield...")
        print(f"Sessions to align: {list(self.session_names)}")

        for i, session_name in enumerate(self.session_names):
            print(f"\nAligning session {session_name}...")
            params = self.align_session_to_widefield(i, slider_limit_px=slider_limit_px)
            wf_transformations[session_name] = params

        return wf_transformations

    def export_alignment_parameters(self):

        aligner_params = {
            'wf_maps': self.widefield_maps,
            'areas_verts': self.areas_verts,
            'wf_transformations': self.wf_transformations
        }

        with open(self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl', 'wb') as f:
            pickle.dump(aligner_params, f)
            print(f"Alignment parameters saved to: {self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl'}")

        return aligner_params

    def import_alignment_parameters(self):

        with open(self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl', 'rb') as f:
            aligner_params = pickle.load(f)
            
        self.widefield_maps = aligner_params['wf_maps']
        self.areas_verts = aligner_params['areas_verts']
        self.wf_transformations = aligner_params['wf_transformations']

        return aligner_params

class WidefieldMapper:

    def __init__(self, aligner_params):

        self.aligner_params = aligner_params
        self.processed_data_dir = r"D:\data-distance-coding-project\processed-data"

    def load_session_df(self, session_name):

        session_data_dir = os.path.join(self.processed_data_dir, session_name)

        df_name = session_name + '_dataframe.pkl'
        df_path = os.path.join(session_data_dir, df_name)

        try:
            with open(df_path, 'rb') as f:
                session_df = pickle.load(f)
                print(f"Session dataframe loaded from {df_name}")
            return session_df
        except Exception as e:
            print(f"Error loading session dataframe: {e}")
            return None

    def load_suite2p_stat(self, session_name):
        
        session_data_dir = os.path.join(self.processed_data_dir, session_name)
        suite2p_stat_path = os.path.join(session_data_dir, 'suite2p', 'plane0', 'stat.npy')

        try:
            suite2p_stat = np.load(suite2p_stat_path, allow_pickle=True)
            print(f"Suite2p stat loaded from {suite2p_stat_path}")
            return suite2p_stat
        except Exception as e:
            print(f"Error loading suite2p stat: {e}")
            return None

    def get_original_coords(self, suite2p_stat, cell_id):
        
        xpix = suite2p_stat[cell_id]['xpix'][~suite2p_stat[cell_id]['overlap']].mean()
        ypix = suite2p_stat[cell_id]['ypix'][~suite2p_stat[cell_id]['overlap']].mean()
        return xpix, ypix

    def transform_coords_to_widefield(self, session_name, xpix, ypix):

        scale = self.aligner_params['wf_transformations'][session_name]['scale']
        rotation = self.aligner_params['wf_transformations'][session_name]['rotation']
        dx = self.aligner_params['wf_transformations'][session_name]['dx']
        dy = self.aligner_params['wf_transformations'][session_name]['dy']

        h_orig, w_orig = 760, 760  # Original 2P image dimensions
        h_wf, w_wf = 400, 400  # Widefield image dimensions

        # Step 2: Resize coordinates from 760x760 to 400x400 space
        x_resized = xpix * (w_wf / w_orig)
        y_resized = ypix * (h_wf / h_orig)

        # Step 3: Apply the SAME transformation matrix that cv2 uses
        center = (w_wf / 2, h_wf / 2)
        
        # Create the rotation+scale matrix (same as cv2.getRotationMatrix2D)
        M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy
        
        # Apply the transformation matrix to the point
        # M * [x, y, 1]^T = [x', y']^T
        point = np.array([x_resized, y_resized, 1])
        transformed_point = M_rot @ point
        
        xwf, ywf = transformed_point[0], transformed_point[1]

        return xwf, ywf

    def assign_area_to_cell(self, xwf, ywf):

        for area_name in self.aligner_params['areas_verts'].keys():
            path = Path(self.aligner_params['areas_verts'][area_name])
            if path.contains_point((xwf, ywf)):
                return area_name
        return 'None'
    
    def plot_cell_in_widefield(self, session_name, cell_id):

        """
        Plot a single cell in both its original session view and transformed to widefield.
        
        Parameters
        ----------
        cell_id : int
            Cell index to plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # LEFT PLOT: Original session
        # Load ops to get mean image
        session_data_dir = os.path.join(self.processed_data_dir, session_name)
        suite2p_ops_path = os.path.join(session_data_dir, 'suite2p', 'plane0', 'ops.npy')
        ops = np.load(suite2p_ops_path, allow_pickle=True).item()
        mean_img = ops['meanImg']
        
        ax1.imshow(mean_img, cmap='gray', extent=[0, 760, 760, 0], origin='upper')
        
        # Get raw pixel coordinates (no transformation needed)
        suite2p_stat = self.load_suite2p_stat(session_name)
        xpix_raw, ypix_raw = self.get_original_coords(suite2p_stat, cell_id)
        
        ax1.scatter(xpix_raw, ypix_raw, c='red', s=100, 
                   alpha=0.7, edgecolors='white', linewidths=1.5,
                   label=f'Cell {cell_id}')
        
        ax1.set_title('Original Session View')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.legend(loc='upper right', fontsize=8)
        
        # RIGHT PLOT: Widefield composite
        widefield = self.aligner_params['wf_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Transform and overlay session
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
        mean_img_resized = cv2.resize(mean_img_norm, (widefield.shape[1], widefield.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        
        h, w = mean_img_resized.shape[:2]
        center = (w / 2, h / 2)
        
        # Get alignment parameters from self.alignment_params
        M_rot = cv2.getRotationMatrix2D(center, self.aligner_params['wf_transformations'][session_name]['rotation'], 
                                         self.aligner_params['wf_transformations'][session_name]['scale'])
        M_rot[0, 2] += self.aligner_params['wf_transformations'][session_name]['dx']
        M_rot[1, 2] += self.aligner_params['wf_transformations'][session_name]['dy']
        
        transformed = cv2.warpAffine(mean_img_resized, M_rot, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Blend session in green
        alpha = 0.5
        session_green = np.zeros_like(composite)
        session_green[:, :, 1] = transformed
        mask = (transformed > 0).astype(float)
        composite = composite * (1 - alpha * mask[:, :, None]) + session_green * alpha
        
        ax2.imshow(composite, extent=[0, 400, 400, 0], origin='upper')
        
        # Transform and plot the cell
        xwf, ywf = self.transform_coords_to_widefield(session_name, xpix_raw, ypix_raw)
        
        ax2.scatter(xwf, ywf, c='red', s=100, 
                   alpha=0.7, edgecolors='white', linewidths=1.5,
                   label=f'Cell {cell_id}')
        
        ax2.set_title('Widefield View with Transformed Cell')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)
    
    def append_widefield_to_session_df(self, session_name):

        """
        Append widefield coordinates and corresponding azimuth/altitude values to session_df.
        
        For each cell in session_df, transforms its coordinates to widefield space and 
        retrieves the azimuth and altitude values at those coordinates.
        """

        session_df = self.load_session_df(session_name)
        suite2p_stat = self.load_suite2p_stat(session_name)

        # Initialize new columns
        widefield_coords = []
        azimuth_values = []
        altitude_values = []
        area_names = []
        
        # Get widefield maps
        azi_map = self.aligner_params['wf_maps']['azi']
        alt_map = self.aligner_params['wf_maps']['alt']
        
        # Process each cell
        for idx, row in session_df.iterrows():

            cell_id = row['cell_id']
            xpix, ypix = self.get_original_coords(suite2p_stat, cell_id)

            if np.isnan(xpix) or np.isnan(ypix):
                print(f"Cell ID: {cell_id} has no original coords")
                #remove the row from the dataframe
                session_df = session_df.drop(idx)
                continue
            
            # Transform to widefield coordinates
            xwf, ywf = self.transform_coords_to_widefield(session_name, xpix, ypix)

            azi_val = azi_map[int(ywf), int(xwf)]
            alt_val = alt_map[int(ywf), int(xwf)]
            
            widefield_coords.append((float(xwf), float(ywf)))
            azimuth_values.append(azi_val)
            altitude_values.append(alt_val)

            area_name = self.assign_area_to_cell(xwf, ywf)
            area_names.append(area_name)
        
        # Append to dataframe
        session_df['wf_coords'] = widefield_coords
        session_df['azi'] = azimuth_values
        session_df['alt'] = altitude_values
        session_df['area'] = area_names

        print(f"Added widefield coordinates and retinotopy values for {len(session_df)} cells")

        session_data_dir = os.path.join(self.processed_data_dir, session_name)
        df_name = session_name + '_dataframe_registered.pkl'
        df_path = os.path.join(session_data_dir, df_name)
        with open(df_path, 'wb') as f:
            pickle.dump(session_df, f)
            print(f"Session dataframe saved to {df_name}")

        return session_df

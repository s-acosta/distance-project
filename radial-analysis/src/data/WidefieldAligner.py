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
            print("No alignment parameters found. Loading areas from retinotopy_areas.pkl and aligning sessions.")
            self.areas_verts = self.load_areas_from_pkl()
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

        snap_mat_path = os.path.join(self.multi_session_dir, 'snap.mat')
        retino_path = os.path.join(self.multi_session_dir, 'retinotopy.npz')
        retino_data = np.load(retino_path)

        if os.path.exists(snap_mat_path):
            snap_data = loadmat(snap_mat_path)
            widefield_img = snap_data['snap']
        else:
            widefield_img = retino_data['f0']
        widefield_img = cv2.resize(widefield_img, (400, 400))
        widefield_maps['img'] = widefield_img

        vfs = cv2.resize(retino_data['vfs'], (400, 400))
        alt = cv2.resize(retino_data['altitude_map'], (400, 400))
        azi = cv2.resize(retino_data['azimuth_map'], (400, 400))
        widefield_maps['VFS'] = gaussian_filter(vfs, sigma=1.5)
        widefield_maps['alt'] = gaussian_filter(alt, sigma=1.5)
        widefield_maps['azi'] = gaussian_filter(azi, sigma=1.5)

        return widefield_maps

    def load_areas_from_pkl(self):

        areas_pkl_path = os.path.join(self.multi_session_dir, 'retinotopy_areas.pkl')
        with open(areas_pkl_path, 'rb') as f:
            areas_verts = pickle.load(f)

        retino = np.load(os.path.join(self.multi_session_dir, 'retinotopy.npz'))
        vfs_h, vfs_w = retino['vfs'].shape
        scale_x = 400.0 / vfs_w
        scale_y = 400.0 / vfs_h

        scaled = {}
        for area, verts in areas_verts.items():
            scaled[area] = [(x * scale_x, y * scale_y) for x, y in verts]
        return scaled

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

        self.areas_verts = areas_verts

        # If any areas already have verts (e.g., from a previous run), show them
        if areas_verts:
            draw_all_patches()

        plt.show()

        return self.areas_verts

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

        # Use existing values if available, otherwise use defaults
        session_name = self.session_names[session_id]
        if hasattr(self, 'wf_transformations') and session_name in self.wf_transformations:
            existing = self.wf_transformations[session_name]
            scale0 = existing['scale']
            rotation0 = existing['rotation']
            dx0 = existing['dx']
            dy0 = existing['dy']
            print(f"Using existing alignment as starting point for {session_name}")
        else:
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

        def on_key(event):
            step = 1
            if event.key == 'right':
                slider_dx.set_val(np.clip(slider_dx.val + step, -slider_limit_px, slider_limit_px))
            elif event.key == 'left':
                slider_dx.set_val(np.clip(slider_dx.val - step, -slider_limit_px, slider_limit_px))
            elif event.key == 'up':
                slider_dy.set_val(np.clip(slider_dy.val - step, -slider_limit_px, slider_limit_px))
            elif event.key == 'down':
                slider_dy.set_val(np.clip(slider_dy.val + step, -slider_limit_px, slider_limit_px))

        # Connect sliders to update function
        slider_scale.on_changed(update)
        slider_rotation.on_changed(update)
        slider_dx.on_changed(update)
        slider_dy.on_changed(update)
        slider_alpha.on_changed(update)
        fig.canvas.mpl_connect('key_press_event', on_key)

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

    def redraw_areas(self):
        """
        Redo area drawing while preserving existing session alignments.
        """
        self.areas_verts = self.draw_areas()
        self.export_alignment_parameters()
        print("Areas redrawn and saved.")

    def redraw_area(self, area_name):
        """
        Redo lasso drawing for a single area.

        Parameters
        ----------
        area_name : str
            Name of the area to redraw (e.g., 'V1', 'LM', 'AL', 'RL', 'AM', 'PM')
        """
        valid_areas = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']
        if area_name not in valid_areas:
            raise ValueError(f"Invalid area name: {area_name}. Must be one of {valid_areas}")

        VFS_map = self.widefield_maps['VFS']

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(VFS_map, cmap='jet', extent=[0, 400, 400, 0], origin='upper')
        ax.set_xlabel("X (units)")
        ax.set_ylabel("Y (units)")

        # Draw existing areas
        for i, (name, verts) in enumerate(self.areas_verts.items()):
            color = 'gray' if name != area_name else 'black'
            linestyle = '--' if name != area_name else '-'
            patch = Polygon(verts, closed=True, facecolor='none',
                          edgecolor=color, linewidth=2, linestyle=linestyle, label=name)
            ax.add_patch(patch)

        ax.legend(loc='upper right')
        ax.set_title(f"Draw new lasso for: {area_name}")
        print(f"Draw new lasso for: {area_name}")

        new_verts = [None]

        def onselect(verts):
            new_verts[0] = list(verts)
            patch = Polygon(verts, closed=True, facecolor='none',
                          edgecolor='red', linewidth=3, label=f'{area_name} (new)')
            ax.add_patch(patch)
            ax.set_title(f"{area_name} redrawn. Close window to save.")
            fig.canvas.draw_idle()
            print(f"{area_name} redrawn. Close the figure window to save.")

        lasso = LassoSelector(ax, onselect, props=dict(color='red', linewidth=3))
        plt.show()

        if new_verts[0] is not None:
            self.areas_verts[area_name] = new_verts[0]
            self.export_alignment_parameters()
            print(f"Area {area_name} updated and saved.")
        else:
            print("No new vertices drawn. Area unchanged.")

    def realign_session(self, session_name):
        """
        Redo alignment for a single session while preserving other sessions.

        Parameters
        ----------
        session_name : str
            Name of the session to realign
        """
        if session_name not in self.session_names:
            raise ValueError(f"Session {session_name} not found. Available: {self.session_names}")

        session_idx = self.session_names.index(session_name)
        print(f"Realigning session: {session_name}")

        params = self.align_session_to_widefield(session_idx)
        self.wf_transformations[session_name] = params
        self.export_alignment_parameters()
        print(f"Session {session_name} realigned and saved.")

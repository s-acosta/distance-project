import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from matplotlib.path import Path


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

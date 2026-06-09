import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import re
import warnings
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d


class VideoVisualizer:
    """
    Creates videos from raw OME-TIFF imaging frames for individual trials.

    Frames must be named with a 6-digit 1-based frame ID suffix, e.g.:
        260326_SAM_SAM044_distance_normal-001_Cycle00001_Ch2_000001.ome.tiff

    The 1-based frame ID corresponds to suite2p frame index + 1, which aligns
    with device.frames_time indices.

    Parameters
    ----------
    device : RadialDevice
        Provides trial timing (off_start, time_per_trial) and frames_time.
    frames_dir : str
        Directory containing the .ome.tiff frame files.
    """

    def __init__(self, device, frames_dir):
        self.device = device
        self.frames_dir = frames_dir
        self._frame_index = None

    def _build_frame_index(self):
        """Build dict mapping 1-based frame ID (int) -> absolute file path (recursive search)."""
        pattern = re.compile(r'_(\d{6})\.ome\.tiff?$')
        index = {}
        for root, _, files in os.walk(self.frames_dir):
            for fname in files:
                m = pattern.search(fname)
                if m:
                    frame_id = int(m.group(1))
                    index[frame_id] = os.path.join(root, fname)
        if not index:
            raise FileNotFoundError(
                f"No .ome.tiff files with 6-digit frame IDs found in: {self.frames_dir}"
            )
        print(f"Indexed {len(index)} frames (IDs {min(index)}..{max(index)})")
        self._frame_index = index

    def _get_trial_frame_ids(self, trial_idx):
        """
        Return sorted array of 1-based TIFF frame IDs for the full trial window.

        Trial window: off_start[trial_idx] to off_start[trial_idx] + time_per_trial.

        Parameters
        ----------
        trial_idx : int
            Index into device.off_start.

        Returns
        -------
        np.ndarray of int
        """
        start_ms = self.device.off_start[trial_idx] * 1000
        end_ms = (self.device.off_start[trial_idx] + self.device.time_per_trial) * 1000
        mask = (self.device.frames_time >= start_ms) & (self.device.frames_time <= end_ms)
        frame_ids = np.where(mask)[0] + 1  # 0-based suite2p index → 1-based TIFF ID
        return frame_ids

    def _load_frames(self, frame_ids):
        """
        Load a sequence of OME-TIFF frames from disk.

        Parameters
        ----------
        frame_ids : array-like of int
            1-based frame IDs to load.

        Returns
        -------
        np.ndarray, shape (n_frames, H, W), dtype float32
        """
        if self._frame_index is None:
            self._build_frame_index()

        frames = []
        missing = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fid in frame_ids:
                if fid not in self._frame_index:
                    missing.append(fid)
                    continue
                img = tifffile.imread(self._frame_index[fid]).astype(np.float32)
                if img.ndim == 3:
                    img = img[0]  # multi-page OME: take first page
                frames.append(img)

        if missing:
            print(f"Warning: {len(missing)} frame IDs not found on disk: {missing[:5]}")

        if not frames:
            raise RuntimeError("No frames could be loaded for the requested trial.")

        return np.stack(frames, axis=0)

    def _smooth_frames(self, stack, gaussian_frames=1.5):
        return gaussian_filter1d(stack, sigma=gaussian_frames, axis=0)

    def _rolling_average(self, stack, window=8):
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(stack, size=window, axis=0)

    def _apply_colormap(self, stack_norm, cmap, cmap_range):
        """
        Apply a colormap to a [0,1] normalized stack, sampling only a sub-range
        of the colormap to control saturation.

        Parameters
        ----------
        stack_norm : np.ndarray, shape (T, H, W), values in [0, 1]
        cmap : str
            Matplotlib colormap name.
        cmap_range : tuple of (float, float)
            (lo, hi) portion of the colormap to sample, both in [0, 1].
            e.g. (0.0, 0.6) avoids the darkest most-saturated end of 'Greens'.

        Returns
        -------
        np.ndarray, shape (T, H, W, 3), dtype uint8
        """
        lo, hi = cmap_range
        mapped = lo + stack_norm * (hi - lo)
        rgba = cm.get_cmap(cmap)(mapped)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _normalize_frames(self, stack, pmin=2, pmax=98):
        """
        Normalize stack to [0, 1] using percentile clipping.

        Percentiles computed across all frames for consistent color scale.

        Parameters
        ----------
        stack : np.ndarray, shape (T, H, W)
        pmin, pmax : float

        Returns
        -------
        np.ndarray, shape (T, H, W), dtype float32, values in [0, 1]
        """
        lo = np.percentile(stack, pmin)
        hi = np.percentile(stack, pmax)
        if hi == lo:
            return np.zeros_like(stack)
        clipped = np.clip(stack, lo, hi)
        return ((clipped - lo) / (hi - lo)).astype(np.float32)

    def make_trial_video(self, trial_idx, output_path,
                         gaussian_frames=0, window_frames=8, fps=7, cmap='gray',
                         cmap_range=(0.0, 0.85), pmin=2, pmax=98):
        """
        Extract on-period frames for a trial, apply Gaussian smoothing, and save as GIF.

        Parameters
        ----------
        trial_idx : int
            Index into device.on_start / device.on_end.
        output_path : str
            Output file path (must end in .gif).
        gaussian_frames : float, optional
            Gaussian kernel std in frame units. Default 1.5 (= 150 ms at 10 Hz).
        fps : int, optional
            Output frames per second. Default 10 (real-time).
        cmap : str, optional
            Matplotlib colormap. Default 'gray'.
        cmap_range : tuple of (float, float), optional
            Sub-range of colormap to sample, controls saturation. Default (0.0, 0.85).
        pmin, pmax : float, optional
            Percentile bounds for contrast normalization. Default 2/98.

        Returns
        -------
        str
            Path to the saved GIF.
        """
        frame_ids = self._get_on_frame_ids(trial_idx)
        if len(frame_ids) == 0:
            raise ValueError(f"No frames found for trial {trial_idx}.")
        print(f"Trial {trial_idx}: {len(frame_ids)} on-period frames (IDs {frame_ids[0]}..{frame_ids[-1]})")

        stack = self._load_frames(frame_ids)

        if window_frames > 0:
            stack = self._rolling_average(stack, window=window_frames)
        elif gaussian_frames > 0:
            stack = self._smooth_frames(stack, gaussian_frames=gaussian_frames)

        stack_norm = self._normalize_frames(stack, pmin=pmin, pmax=pmax)
        rgb = self._apply_colormap(stack_norm, cmap, cmap_range)

        angles = np.linspace(-30, 60, len(frame_ids))
        angle_labels = [f'{a:.1f}°' for a in angles]

        distance_map = {0: '10cm', 1: '16cm', 2: '22cm', 3: '28cm', 4: 'control'}
        dist_idx = int(self.device.distance_sequence[trial_idx])
        dist_label = distance_map.get(dist_idx, f'd={dist_idx}')

        self._write_gif(rgb, angle_labels, output_path, fps, label=dist_label)
        return output_path

    def _get_on_frame_ids(self, trial_idx):
        """Return 1-based TIFF frame IDs for the stimulus-on period only, truncated to on_length."""
        on_start, on_end = self.device.get_on_times(trial_idx)
        mask = (self.device.frames_time > on_start) & (self.device.frames_time < on_end)
        suite2p_indices = np.where(mask)[0][:self.device.on_length]
        return suite2p_indices + 1  # 1-based TIFF IDs

    def preview_trial(self, trial_idx, gaussian_frames=1.5, fps=10, cmap='Greens',
                      cmap_range=(0.0, 0.6), pmin=2, pmax=98):
        """
        Preview the stimulus-on period of a trial inline (Jupyter) without saving to disk.
        Displays the current visual angle in the upper-right corner.

        Parameters
        ----------
        trial_idx : int
            Index into device.on_start / device.on_end.
        gaussian_frames : float, optional
            Gaussian kernel std in frame units. Default 1.5 (= 150 ms at 10 Hz).
        fps : int, optional
            Playback speed in frames per second. Default 10 (real-time).
        cmap : str, optional
            Matplotlib colormap. Default 'gray'.
        cmap_range : tuple of (float, float), optional
            Sub-range of colormap to sample, controls saturation. Default (0.0, 0.85).
        pmin, pmax : float, optional
            Percentile bounds for contrast normalization. Default 2/98.
        """
        frame_ids = self._get_on_frame_ids(trial_idx)
        if len(frame_ids) == 0:
            raise ValueError(f"No on-period frames found for trial {trial_idx}.")
        print(f"Trial {trial_idx}: {len(frame_ids)} on-period frames (IDs {frame_ids[0]}..{frame_ids[-1]})")

        stack = self._load_frames(frame_ids)

        if gaussian_frames > 0:
            stack = self._smooth_frames(stack, gaussian_frames=gaussian_frames)

        stack_norm = self._normalize_frames(stack, pmin=pmin, pmax=pmax)
        rgb = self._apply_colormap(stack_norm, cmap, cmap_range)

        angle_range = np.array(self.device.angle_range).flatten()
        angle_labels = [f'{a:.1f}°' for a in angle_range]

        import tempfile
        from IPython.display import Image, display

        tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
        tmp.close()
        self._write_gif(rgb, angle_labels, tmp.name, fps)
        display(Image(filename=tmp.name))
        return tmp.name

    def _write_gif(self, rgb, angle_labels, output_path, fps, label=''):
        """Write an RGB frame stack with angle and distance overlay. Supports .gif and .mp4.

        Parameters
        ----------
        angle_labels : list of str
            Per-frame angle strings.
        label : str
            Static label shown below the angle (e.g. distance). Same every frame.
        """
        ext = os.path.splitext(output_path)[1].lower()

        if ext == '.mp4':
            self._write_mp4(rgb, angle_labels, output_path, fps, label=label)
        else:
            from matplotlib.animation import PillowWriter

            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.axis('off')

            im = ax.imshow(rgb[0], vmin=0, vmax=255, animated=True)
            angle_text = ax.text(
                0.98, 0.96, f'angle: {angle_labels[0]}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=13, fontweight='bold', color='white', animated=True
            )
            dist_text = ax.text(
                0.98, 0.88, f'distance: {label}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=13, color='white', animated=True
            )

            def update(i):
                im.set_data(rgb[i])
                angle_text.set_text(f'angle: {angle_labels[i]}')
                return im, angle_text, dist_text

            anim = animation.FuncAnimation(fig, update, frames=len(rgb),
                                           interval=1000/fps, blit=True)
            anim.save(output_path, writer=PillowWriter(fps=fps))
            plt.close(fig)

        print(f"Saved: {output_path}")

    def _write_mp4(self, rgb, angle_labels, output_path, fps, label=''):
        """Write frames as MP4 using cv2, with labels burned in via matplotlib."""
        import cv2

        H, W = rgb.shape[1], rgb.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('black')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        im = ax.imshow(rgb[0], vmin=0, vmax=255)
        angle_text = ax.text(
            0.98, 0.96, f'angle: {angle_labels[0]}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=max(8, H // 35), fontweight='bold', color='white'
        )
        ax.text(
            0.98, 0.88, f'distance: {label}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=max(8, H // 35), color='white'
        )

        canvas = fig.canvas
        canvas.draw()

        for i in range(len(rgb)):
            im.set_data(rgb[i])
            angle_text.set_text(f'angle: {angle_labels[i]}')
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

        writer.release()
        plt.close(fig)

    def make_average_video(self, output_path, monocular=None,
                           gaussian_frames=1.5, fps=7, cmap='gray',
                           cmap_range=(0.0, 0.85), pmin=2, pmax=98):
        """
        Average frames across all non-control trials at each time point and save as GIF.

        Parameters
        ----------
        output_path : str
            Output file path (must end in .gif).
        monocular : int or None, optional
            Filter by eye condition (0=binocular, 1=monocular). None = all. Default None.
        gaussian_frames : float, optional
            Gaussian kernel std in frame units. Default 1.5.
        fps : int, optional
            Output frames per second. Default 7.
        cmap : str, optional
            Matplotlib colormap. Default 'gray'.
        cmap_range : tuple of (float, float), optional
            Sub-range of colormap to sample. Default (0.0, 0.85).
        pmin, pmax : float, optional
            Percentile bounds for contrast normalization. Default 2/98.
        """
        trial_indices = self.device.get_condition_trials(monocular=monocular)
        print(f"Averaging {len(trial_indices)} trials...")

        all_stacks = []
        for trial_idx in trial_indices:
            frame_ids = self._get_on_frame_ids(trial_idx)
            stack = self._load_frames(frame_ids)   # (on_length, H, W)
            all_stacks.append(stack)

        # Average across trials: (n_trials, on_length, H, W) -> (on_length, H, W)
        mean_stack = np.mean(np.stack(all_stacks, axis=0), axis=0)

        if gaussian_frames > 0:
            mean_stack = self._smooth_frames(mean_stack, gaussian_frames=gaussian_frames)

        stack_norm = self._normalize_frames(mean_stack, pmin=pmin, pmax=pmax)
        rgb = self._apply_colormap(stack_norm, cmap, cmap_range)

        angle_range = np.array(self.device.angle_range).flatten()
        self._write_gif(rgb, angle_range, output_path, fps)
        return output_path

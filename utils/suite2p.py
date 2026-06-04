import os
import shutil
import numpy as np


def split_suite2p_recordings(joint_plane0_dir, output_dirs, frame_counts):
    """
    Split a combined suite2p plane0 directory into N separate directories,
    one per recording, based on explicit frame counts.

    Args:
        joint_plane0_dir: Path to the combined suite2p/plane0 directory
        output_dirs:      List of output directory paths, one per recording
        frame_counts:     List of ints, number of frames per recording
                          (must sum to the total number of frames in F.npy)
    """
    if len(output_dirs) != len(frame_counts):
        raise ValueError("output_dirs and frame_counts must have the same length")

    raw_f = np.load(os.path.join(joint_plane0_dir, 'F.npy'))
    total_frames = raw_f.shape[1]

    if sum(frame_counts) != total_frames:
        raise ValueError(
            f"frame_counts sum to {sum(frame_counts)} but F.npy has {total_frames} frames"
        )

    # Preload all frame-based arrays
    f_neu = np.load(os.path.join(joint_plane0_dir, 'Fneu.npy'))
    spks  = np.load(os.path.join(joint_plane0_dir, 'spks.npy'))

    frames_time_path = os.path.join(joint_plane0_dir, 'frames_time.npy')
    frames_time = np.load(frames_time_path) if os.path.isfile(frames_time_path) else None

    dff_path = os.path.join(joint_plane0_dir, 'dFF.npy')
    dff = np.load(dff_path) if os.path.isfile(dff_path) else None

    # Compute slice boundaries from cumulative frame counts
    boundaries = np.concatenate([[0], np.cumsum(frame_counts)])

    for i, out_dir in enumerate(output_dirs):
        os.makedirs(out_dir, exist_ok=True)
        start, end = boundaries[i], boundaries[i + 1]

        np.save(os.path.join(out_dir, 'F.npy'),    raw_f[:, start:end])
        np.save(os.path.join(out_dir, 'Fneu.npy'), f_neu[:, start:end])
        np.save(os.path.join(out_dir, 'spks.npy'), spks[:, start:end])

        if frames_time is not None:
            np.save(os.path.join(out_dir, 'frames_time.npy'), frames_time[start:end])

        if dff is not None:
            np.save(os.path.join(out_dir, 'dFF.npy'), dff[:, start:end])

        # Cell metadata is the same for all recordings — copy unchanged
        for fname in ['iscell.npy', 'stat.npy', 'ops.npy', 'redcell.npy']:
            src = os.path.join(joint_plane0_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_dir, fname))

        print(f"Recording {i + 1}: {frame_counts[i]} frames → {out_dir}")

    print(f"Done. {raw_f.shape[0]} cells split into {len(output_dirs)} recordings.")

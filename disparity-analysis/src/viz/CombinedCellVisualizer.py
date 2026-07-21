import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import setup_plot_style
from src.viz.CellVisualizer import CellVisualizer
from src.viz.DisparityCellVisualizer import DisparityCellVisualizer

setup_plot_style()


class CombinedCellVisualizer:
    """
    Per-cell figure combining the two goggles experiments for the same neuron:
    the monocular left/right-eye comparison on top and the binocular disparity
    plot below.

    The two sessions must come from the same suite2p run (identical
    ``suite2p_id``) so a cell id refers to the same neuron in both.

    Examples
    --------
    >>> from src.data.MonocularDevice import MonocularDevice
    >>> from src.data.DisparityDevice import DisparityDevice
    >>> from src.viz.CombinedCellVisualizer import CombinedCellVisualizer
    >>> M = MonocularDevice(mouse_name='SAM045', record_date='260713',
    ...                     experiment_name='goggles-monocular')
    >>> B = DisparityDevice(mouse_name='SAM045', record_date='260713',
    ...                         experiment_name='goggles-disparity')
    >>> CombinedCellVisualizer(M, B).make_combined_plot(239)
    """

    def __init__(self, mono_device, disp_device):
        if not np.array_equal(mono_device.suite2p_id, disp_device.suite2p_id):
            print("[warn] monocular and disparity sessions have different suite2p_id; "
                  "a cell id may not refer to the same neuron in both.")
        self.mono = CellVisualizer(mono_device)
        self.disp = DisparityCellVisualizer(disp_device)

    def make_combined_plot(self, cell, units='deg'):
        """
        One figure: monocular comparison (top) + disparity plot (bottom).

        ``units`` ('deg' or 'px') is forwarded to the disparity overlays.
        """
        n = self.disp.device.n_disparities
        fig = plt.figure(figsize=(4.8 * n, 17), constrained_layout=True)
        top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.4], hspace=0.12)

        self.mono.make_comparison_plot(cell, fig=top)
        top.suptitle('Monocular  —  left vs right eye', fontsize=30, fontweight='bold')

        self.disp.make_disparity_plot(cell, units=units, fig=bottom)
        bottom.suptitle('Binocular disparity', fontsize=30, fontweight='bold')

        fig.suptitle(f'Cell {cell}', fontsize=24)
        return fig

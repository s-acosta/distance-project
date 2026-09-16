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

    def _mono_rf_px(self, cell):
        """The cell's two monocular RF positions, in signed screen pixels."""
        dev = self.mono.device
        cond = self.mono._cond()
        li, _ = dev.find_rf_position(cell, eye='left', **cond)
        ri, _ = dev.find_rf_position(cell, eye='right', **cond)
        return (float(dev.get_signed_screen_range('left')[li]),
                float(dev.get_signed_screen_range('right')[ri]))

    def make_combined_plot(self, cell, units='px', fontscale=None):
        """
        One figure: monocular comparison (top) + disparity plot (bottom).

        In ``units='px'`` (the default) both halves share the monocular session's
        signed pixel axis and the monocular RF positions are marked on the
        disparity panel, so the two can be read down a single column: the
        disparity whose left curve reaches the left RF while its right curve sits
        on the right RF is the cell's preferred one. ``units='deg'`` gives the
        disparity half its own per-eye angle panels instead.

        ``fontscale`` multiplies every font size in *both* halves. This figure is
        several times wider than either plot is on its own, so whatever it is
        displayed in shrinks it to fit and text sized for a standalone plot ends
        up unreadable; scaling the type up is the only lever, since shrinking the
        canvas instead makes matplotlib drop the constrained layout. Left as None
        it picks a value per mode: 'deg' packs three panels into the bottom row
        where 'px' packs two, so it runs out of room far sooner and overlaps past
        about 1.1. Push it higher by hand in 'px', which has room to spare.
        """
        if fontscale is None:
            fontscale = 2.0 if units == 'px' else 1.1

        n = self.disp.device.n_disparities
        fig = plt.figure(figsize=(4.8 * n, 17), constrained_layout=True)
        top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.4], hspace=0.12)

        # the monocular plot's own sizes are already ~40% larger than the disparity
        # plot's, so the same multiplier would leave it proportionally bigger; this
        # lands the two halves on matching type
        self.mono.make_comparison_plot(cell, fig=top, fontscale=fontscale * 0.7)
        # the three headings are already large; scaling them too makes them collide
        top.suptitle('Monocular  —  left vs right eye', fontsize=34, fontweight='bold')

        # share the monocular pixel axis so the two halves line up column-wise
        dev = self.mono.device
        xlim = mono_rf = None
        if units == 'px':
            xlim = (dev.get_signed_screen_range('left').min(),
                    dev.get_signed_screen_range('right').max())
            mono_rf = self._mono_rf_px(cell)

        self.disp.make_disparity_plot(cell, units=units, fig=bottom,
                                      mono_rf=mono_rf, xlim=xlim, fontscale=fontscale)
        bottom.suptitle('Binocular disparity', fontsize=34, fontweight='bold')

        fig.suptitle(f'Cell {cell}', fontsize=30)
        return fig

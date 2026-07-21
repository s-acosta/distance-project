import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Palette matching the distances in radial-analysis (distance-analysis):
# orange, green, blue, dark-purple, grey (grey = control).
DISTANCE_COLORS = ['#FFA500', '#35b779', '#31688e', '#440154', 'grey']

# Continuous distance scale (the real distances, no grey control), for sampling
# an arbitrary number of evenly-spaced colors.
DISTANCE_CMAP = LinearSegmentedColormap.from_list('distance', DISTANCE_COLORS[:-1])


def distance_palette(n):
    """``n`` colors sampled evenly along the distance colormap scale."""
    return [DISTANCE_CMAP(x) for x in np.linspace(0, 1, n)]


class CellVizBase:
    """Shared helpers for per-cell visualizers (eye and disparity)."""

    def __init__(self, device):
        self.device = device

    def _vlimits(self, cell):
        """(vmin, vmax) color limits from the 1st/99th percentile of a cell's response."""
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        vmin = np.percentile(self.device.response[idx, :], 1)
        vmax = np.percentile(self.device.response[idx, :], 99)
        return vmin, vmax

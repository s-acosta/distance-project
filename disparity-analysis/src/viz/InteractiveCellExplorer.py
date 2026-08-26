import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import setup_plot_style

setup_plot_style()


class InteractiveCellExplorer:
    """
    One figure combining a session's location map (top) with a per-cell plot
    (bottom) that updates when you click a cell in the map.

    Requires an interactive matplotlib backend that delivers click events —
    put ``%matplotlib widget`` (ipympl) at the top of the notebook cell before
    calling :meth:`show`; the default static inline backend renders a picture,
    not a live widget, so clicks on it go nowhere.

    Parameters
    ----------
    session_viz : SessionVisualizer
        Supplies the location map (``plot_location_map``) and cell positions
        (``device.suite2p_id`` / ``_cell_xy``).
    cell_viz : CellVisualizer (or anything with a compatible method)
        Supplies the per-cell plot. ``cell_plot_method`` must draw into a
        given figure rather than creating its own — exactly what
        ``CellVisualizer.make_comparison_plot(cell, fig=...)`` already does.
    cell_plot_method : str
        Name of the method on ``cell_viz`` to call for the bottom panel.
    map_kwargs : dict, optional
        Extra keyword arguments forwarded to ``session_viz.plot_location_map``
        (e.g. ``{'corr_threshold': 0.5, 'show_unresponsive': False}``).
    pick_radius : float
        Max distance (FOV pixels) from a click to a cell's centroid for it to
        count as a hit. Clicks farther than this from every cell are ignored.

    Examples
    --------
    >>> %matplotlib widget
    >>> from src.viz.SessionVisualizer import SessionVisualizer
    >>> from src.viz.CellVisualizer import CellVisualizer
    >>> from src.viz.InteractiveCellExplorer import InteractiveCellExplorer
    >>> sv = SessionVisualizer(M, stats_df=stats_df, corr_threshold=0.75)
    >>> cv = CellVisualizer(M, stats_df=stats_df)
    >>> explorer = InteractiveCellExplorer(sv, cv)
    >>> explorer.show()          # starts on an arbitrary reliable cell
    >>> explorer.select_cell(348)  # or jump to one programmatically
    """

    def __init__(self, session_viz, cell_viz, cell_plot_method='make_comparison_plot',
                 map_kwargs=None, pick_radius=25):
        self.session_viz = session_viz
        self.cell_viz = cell_viz
        self.cell_plot_method = cell_plot_method
        self.map_kwargs = map_kwargs or {}
        self.pick_radius = pick_radius

        self.fig = None
        self.map_ax = None
        self.cell_fig = None
        self.selected_cell = None
        self._highlight = None
        self._positions = None

    def show(self, cell=None, width=16, dpi=60, cell_plot_aspect=22 / 5, bottom_margin=0.9,
             map_scale=0.88, row_gap=0.4):
        """
        Build the combined figure, wire up click-to-select, and return it.

        Both panels span the same figure ``width`` (inches). The location map
        uses ``imshow`` with a locked (equal) aspect ratio, so its row's height
        is derived from the FOV image's own pixel aspect ratio at that width —
        rather than a guessed height ratio — so the image fills the full width
        with no side padding. The cell-plot row's height is likewise derived
        from ``cell_plot_aspect`` (width:height), which defaults to
        ``CellVisualizer.make_comparison_plot``'s own standalone figsize
        (22, 5), so its four panels keep their usual proportions (at that
        width, comfortably fitting its fixed ~20-28pt fonts without titles
        overlapping) instead of being stretched or squashed to fit an
        arbitrary box.

        ``bottom_margin`` (inches) reserves a blank row below the cell-plot
        row. ``constrained_layout``, for a subfigure nested inside another
        subfigure (as the cell-plot row is here), doesn't reliably reserve
        room for its axes' x-tick labels/axis titles — they can render past
        its bottom edge and get clipped regardless of how much height the row
        itself is given (verified: growing the row's own height barely moves
        the overflow, since ``constrained_layout`` keeps assigning those axes
        ~0 bottom margin either way). A dedicated blank buffer row gives that
        overflow somewhere to land instead. The default comfortably covers
        ``CellVisualizer.make_comparison_plot``'s tick labels; raise it if you
        use a different ``cell_plot_method`` and still see clipping.

        Unlike a static inline plot, an ipympl (``%matplotlib widget``) figure
        renders at its literal pixel size (``width`` inches x ``dpi``) rather
        than being scaled down to fit the notebook cell. Lowering ``dpi``
        shrinks the rendering (canvas and fonts together, same proportions)
        without affecting whether things fit; lowering ``width`` risks
        reintroducing overlap, since the fixed point-sized fonts need enough
        absolute inches per panel regardless of dpi.

        ``map_scale`` shrinks the map row a bit below its full-width size
        (``<1`` trades the "fills the width exactly" property for a smaller,
        centered map with a little side padding). ``row_gap`` (inches) is a
        blank buffer row between the map and the cell plot, the same
        technique ``bottom_margin`` uses — plain height-ratio tweaks don't
        reliably add breathing room between nested subfigures either.
        """
        sv = self.session_viz
        self._positions = {int(c): sv._cell_xy(int(c)) for c in sv.device.suite2p_id}

        if cell is None:
            sets = sv.reliable_sets()
            pool = sets.get('either') or sets.get('reliable') or sets['all']
            cell = sorted(pool)[0]

        img_h, img_w = sv._fov_image().shape[:2]
        map_height = width * (img_h / img_w) * map_scale
        cell_height = width / cell_plot_aspect

        self.fig = plt.figure(
            figsize=(width, map_height + row_gap + cell_height + bottom_margin), dpi=dpi,
            constrained_layout=True)
        top, _gap, self.cell_fig, _spacer = self.fig.subfigures(
            4, 1, height_ratios=[map_height, row_gap, cell_height, bottom_margin])
        self.map_ax = top.subplots(1, 1)

        sv.plot_location_map(ax=self.map_ax, **self.map_kwargs)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self.select_cell(cell)
        return self.fig

    def select_cell(self, cell):
        """Redraw the bottom panel for ``cell`` and highlight it on the map."""
        cell = int(cell)
        if cell not in self._positions:
            raise ValueError(f"cell {cell} is not in this session's suite2p_id.")
        self.selected_cell = cell

        for ax in list(self.cell_fig.axes):     # clear the previous cell's panels
            ax.remove()
        getattr(self.cell_viz, self.cell_plot_method)(cell, fig=self.cell_fig)
        self.cell_fig.suptitle(f'Cell {cell}', fontsize=20, fontweight='bold')

        if self._highlight is not None:
            self._highlight.remove()
        x, y = self._positions[cell]
        self._highlight = self.map_ax.scatter(
            [x], [y], s=280, facecolor='none', edgecolor='yellow', linewidth=2.5, zorder=5,
        )
        self.map_ax.set_title(f'{self.session_viz.device.session_name} — cell {cell} selected')

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.map_ax or event.xdata is None:
            return
        cell = self._nearest_cell(event.xdata, event.ydata)
        if cell is not None:
            self.select_cell(cell)

    def _nearest_cell(self, x, y):
        """Closest cell to (x, y) within ``pick_radius``, or None."""
        cells = np.fromiter(self._positions.keys(), dtype=int)
        xy = np.array(list(self._positions.values()))
        dist = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        i = int(np.argmin(dist))
        return int(cells[i]) if dist[i] <= self.pick_radius else None

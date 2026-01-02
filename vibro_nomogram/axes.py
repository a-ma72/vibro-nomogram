from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.axis import Axis, XTick
from matplotlib.backend_bases import RendererBase
from matplotlib.collections import LineCollection
from matplotlib.text import Annotation
from matplotlib.transforms import Transform

from .transforms import SpecTransform, _eps, _log_twopi


class OrderAxis(Axis):
    """
    Custom Axis class for drawing diagonal grid lines representing different orders.

    This axis is responsible for rendering grid lines and labels for quantities
    that are related to the main axes by a factor: Y = y / (2*pi*f)**order.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The parent axes.
    order : int
        The order of the relationship (e.g., 1 for displacement (integration),
        -1 for acceleration (differentiation) relative to velocity).
    **kwargs
        Additional keyword arguments passed to the Axis constructor.
    """

    axis_name = "order_axis"
    _tick_class = XTick
    transform: SpecTransform
    transData: Transform

    def __init__(self, axes: Axes, order: int, **kwargs) -> None:
        """
        Initialize the OrderAxis.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The parent axes.
        order : int
            The order of the axis.
        **kwargs
            Additional keyword arguments.
        """
        self.order = order
        self.transform = SpecTransform(order=order)
        self.transData = self.transform.inverted() + axes.transData
        super().__init__(axes, **kwargs)
        self._set_defaults()

    def _init(self) -> None:
        """Initialize the axis defaults."""
        # Axis has no attribute _init in base class, so we comment it out
        # super()._init()
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Set default locators, formatters, and grid info."""
        self.set_major_locator(mticker.LogLocator(base=10.0, numticks=15))
        self.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        self.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.2g}"))

        self._grid_info = {
            "major": {
                "visible": False,
                "style": {"color": "0.75", "linestyle": ":", "linewidth": 0.5},
            },
            "minor": {
                "visible": False,
                "style": {"color": "0.75", "linestyle": ":", "linewidth": 0.5},
            },
        }

    def set_default_intervals(self) -> None:
        pass

    def get_view_interval(self) -> tuple[float, float]:
        return 0, 1

    def set_view_interval(self, vmin: float, vmax: float, ignore: bool = False) -> None:
        pass

    def get_minpos(self) -> float:
        return 1e-30

    def get_ylim(self) -> tuple[float, float]:
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        if self.order < 0:
            edges = [(xlim[0], ylim[0]), (xlim[1], ylim[1])]
        else:
            edges = [(xlim[1], ylim[0]), (xlim[0], ylim[1])]
        y_min, y_max = self.transform.transform(edges).T[1]
        return y_min, y_max

    def get_tick_space(self) -> int:
        """
        Estimate the number of ticks that can fit on the axis.

        Returns
        -------
        int
            Estimated number of ticks.
        """
        if self.axes is None:
            return 0
        bbox = self.axes.bbox
        length = (bbox.width**2 + bbox.height**2) ** 0.5
        return int(length / 40)

    def grid(self, visible: bool = None, which: str = "major", **kwargs) -> None:
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None, optional
            Whether to show the grid lines. If None, defaults to True.
        which : {"major", "minor", "both"}, optional
            The grid lines to apply the changes to. Default is "major".
        **kwargs
            Style properties for the grid lines (color, linestyle, etc.).
        """
        if visible is None:
            visible = True

        style_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["color", "linestyle", "linewidth", "alpha"]
        }

        if which in ("major", "both"):
            self._grid_info["major"]["visible"] = visible
            if style_kwargs:
                self._grid_info["major"]["style"].update(style_kwargs)

        if which in ("minor", "both"):
            self._grid_info["minor"]["visible"] = visible
            if style_kwargs:
                self._grid_info["minor"]["style"].update(style_kwargs)

    def get_transform(self) -> SpecTransform:
        return SpecTransform(order=self.order)

    def plot(self, *args, **kwargs) -> list:
        """
        Plot data using the transform associated with this axis order.

        Parameters
        ----------
        *args
            Positional arguments passed to the plot method (e.g., x, y).
        **kwargs
            Keyword arguments passed to the plot method.
        """
        if "transform" not in kwargs:
            # The transform needs to convert from the specific order (e.g., displacement)
            # to the base order (velocity). This is the inverse of the transform
            # that defines the axis (velocity -> displacement).
            kwargs["transform"] = self.transData
        return self.axes.plot(*args, **kwargs)

    def draw(self, renderer: RendererBase, *args, **kwargs) -> None:
        """
        Draw the axis elements (grid lines and labels).

        Parameters
        ----------
        renderer : matplotlib.backend_bases.RendererBase
            The renderer to use for drawing.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        if not self.get_visible():
            return

        if self._grid_info["major"]["visible"]:
            self._draw_order_grid_lines(renderer, "major")
        if self._grid_info["minor"]["visible"]:
            self._draw_order_grid_lines(renderer, "minor")

    def _clip_lines_to_box(
        self,
        m: npt.ArrayLike,
        c: npt.ArrayLike,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool64]]:  # noqa: E501
        """
        Clip lines y = mx + c to the bounding box defined by x/y limits.
        """
        c = np.asarray(c)
        x_from_y_min = (y_min - c) / m
        x_from_y_max = (y_max - c) / m
        x_lim_1 = np.minimum(x_from_y_min, x_from_y_max)
        x_lim_2 = np.maximum(x_from_y_min, x_from_y_max)
        x_start = np.maximum(x_min, x_lim_1)
        x_end = np.minimum(x_max, x_lim_2)
        valid = (x_end - x_start) > _eps
        x_start = x_start[valid]
        x_end = x_end[valid]
        c_valid = c[valid]
        y_start = m * x_start + c_valid
        y_end = m * x_end + c_valid
        p1s = np.column_stack((x_start, y_start))
        p2s = np.column_stack((x_end, y_end))
        return p1s, p2s, valid

    def _draw_order_grid_lines(self, renderer: RendererBase, which: str) -> None:
        """
        Draw the grid lines for the specified tick type.
        """
        # Get current axis limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Helper to safely calculate log10 of limits, handling non-positive values
        def safe_log(lims):
            vmin, vmax = sorted(lims)
            if vmin <= 0:
                if vmax > 0:
                    vmin = vmax * 1e-4
                else:
                    vmin = 1e-30
            return np.log10([vmin, vmax])

        # Calculate log limits for clipping
        log_xlim = safe_log(xlim)
        log_ylim = safe_log(ylim)

        # Select the appropriate locator
        if which == "major":
            locator = self.get_major_locator()
        else:
            locator = self.get_minor_locator()

        # Get tick values in the transformed axis units
        vmin, vmax = self.get_ylim()
        ticks = locator.tick_values(vmin, vmax)

        # Transform tick values back to the constant 'c' for the line equation
        # in log-log space: log(y) = order * log(f) + c
        c_ticks = SpecTransform._log_inverse(
            f=np.ones_like(ticks), y=ticks, order=self.order
        )

        is_log_x = self.axes.get_xscale() == "log"
        is_log_y = self.axes.get_yscale() == "log"

        # Calculate start and end points of the grid lines clipped to the visible area
        p1s, p2s, valid = self._clip_lines_to_box(
            self.order,  # The order is the slope in log-log space
            c_ticks,
            log_xlim[0],
            log_xlim[1],
            log_ylim[0],
            log_ylim[1],
        )
        valid_c_ticks = c_ticks[valid]

        # Generate line segments
        if is_log_x and is_log_y:
            # If both axes are log, the lines are straight segments
            segments = np.stack((10**p1s, 10**p2s), axis=1)
        else:
            # If not log-log, we need to interpolate points to draw curves
            segments = []
            for i in range(len(valid_c_ticks)):
                p1_log, p2_log = p1s[i], p2s[i]
                t = np.linspace(0, 1, 100)
                x_log = p1_log[0] + t * (p2_log[0] - p1_log[0])
                y_log = p1_log[1] + t * (p2_log[1] - p1_log[1])
                segments.append(np.column_stack((10**x_log, 10**y_log)))

        # Draw tick labels if these are major grid lines
        if which == "major":
            self._draw_tick_labels(
                renderer,
                valid_c_ticks,
                p1s,
                p2s,
                log_xlim,
                log_ylim,
                formatter=self.get_major_formatter(),
            )

        # Draw the grid lines using LineCollection
        if len(segments) > 0:
            style = self._grid_info[which]["style"]
            lc = LineCollection(
                segments, transform=self.axes.transData, zorder=1, **style
            )
            lc.draw(renderer)

    def _draw_tick_labels(
        self,
        renderer: RendererBase,
        c_vals: npt.ArrayLike,
        p1s: npt.NDArray[np.float64],
        p2s: npt.NDArray[np.float64],
        log_xlim: tuple[float, float],
        log_ylim: tuple[float, float],
        formatter=None,
    ) -> None:
        """
        Draw tick labels for the grid lines.
        """
        if len(c_vals) == 0:
            return

        # Determine which end of the line segment to label based on order (slope direction)
        label_pos_log, other_pos_log = (p1s, p2s) if self.order < 0 else (p2s, p1s)

        # Calculate a small segment near the label position to determine the angle in display space
        p_start, p_end = label_pos_log, other_pos_log
        p_tangent = p_start + 0.01 * (p_end - p_start)

        # Transform points to display coordinates to calculate screen angles
        p1_disp = self.axes.transData.transform(10**p_start)
        p2_disp = self.axes.transData.transform(10**p_tangent)
        dx, dy = p2_disp[:, 0] - p1_disp[:, 0], p2_disp[:, 1] - p1_disp[:, 1]

        # Calculate rotation angle for the text
        angle = np.rad2deg(np.arctan2(dy, dx))
        angle = np.where(angle > 90, angle - 180, angle)
        angle = np.where(angle < -90, angle + 180, angle)

        # Calculate offset vectors for text placement along the line extension
        offset_vecs = np.stack([dx, dy], axis=1)
        norms = np.linalg.norm(offset_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        offset_vecs = 10 * offset_vecs / norms

        # Determine horizontal alignment based on where the line intersects the bounding box
        has = np.full(len(c_vals), "center", dtype=object)
        x, y = label_pos_log[:, 0], label_pos_log[:, 1]

        # Check intersection with boundaries
        mask_left = np.abs(x - log_xlim[0]) < _eps
        mask_right = np.abs(x - log_xlim[1]) < _eps
        mask_top = (np.abs(y - log_ylim[1]) < _eps) & ~mask_left & ~mask_right

        has[mask_left] = "left"
        has[mask_right] = "right"

        # Adjust offsets for top boundary to avoid overlap
        annotate_offset_vecs = np.where(mask_top[:, None], offset_vecs * 2, offset_vecs)
        has[mask_top] = "left" if self.order < 0 else "right"

        for i, c in enumerate(c_vals):
            # Calculate the actual value for the label
            log_tick_val = c - self.order * _log_twopi
            tick_val = 10**log_tick_val
            txt = formatter(tick_val, i) if formatter else f"{tick_val:.2g}"
            xy_data = (10 ** label_pos_log[i, 0], 10 ** label_pos_log[i, 1])

            # Create and draw the text annotation
            ann = Annotation(
                txt,
                xy=xy_data,
                xytext=tuple(annotate_offset_vecs[i]),
                textcoords="offset points",
                ha=has[i],
                va="baseline",
                fontsize=self.axes.yaxis.get_majorticklabels()[0].get_fontsize() * 0.8,
                rotation=angle[i],
                rotation_mode="anchor",
                clip_on=False,
            )
            ann.set_transform(self.axes.transData)
            ann.axes = self.axes
            ann.set_figure(self.figure)
            ann.draw(renderer)

            # Draw a small tick mark (extension of the grid line)
            tick = Annotation(
                "",
                xy=xy_data,
                xytext=tuple(offset_vecs[i] / 1.2),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="-", color="black", linewidth=1.2, clip_on=True
                ),
                rotation=angle[i],
                rotation_mode="anchor",
                zorder=1,
            )
            tick.set_transform(self.axes.transData)
            tick.axes = self.axes
            tick.set_figure(self.figure)
            tick.draw(renderer)

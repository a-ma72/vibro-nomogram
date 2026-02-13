"""
Matplotlib projection class for frequency-domain vibration analysis.

This module provides a custom Matplotlib projection (`FrequencySpaceAxes`) and
associated classes (`OrderAxis`, `SpecTransform`) to visualize spectral data.
It is particularly useful for vibration analysis where the relationship between
displacement, velocity, and acceleration is frequency-dependent.

The projection allows plotting velocity vs. frequency (log-log) while simultaneously
showing grid lines for constant displacement and constant acceleration.
"""

from __future__ import annotations

__author__ = "Andreas Martin"
__version__ = "0.1.0"


from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase

from .axes import OrderAxis
from .formatting import (
    AccelLogFormatter,
    DisplacementFormatter,
    GravityLogFormatter,
    GravityLogLocator,
)


class FrequencySpaceAxes(Axes):
    """
    Custom Axes for frequency domain plots with diagonal grid lines.

    This axes class sets up a log-log plot for frequency vs. velocity (by default)
    and adds diagonal grid lines for displacement and acceleration.

    Parameters
    ----------
    *args
        Positional arguments passed to Axes.
    xscale : str, optional
        Scale for the x-axis. Default is "log".
    yscale : str, optional
        Scale for the y-axis. Default is "log".
    iaxis : OrderAxis, optional
        Axis for displacement (order = 1).
    daxis : OrderAxis, optional
        Axis for acceleration (order = -1).
    **kwargs
        Keyword arguments passed to Axes.
    """

    name = "frequency_space"
    iaxis: OrderAxis | None = None  # Integration axis (displacement)
    daxis: OrderAxis | None = None  # Differentiation axis (acceleration)

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the FrequencySpaceAxes.

        Parameters
        ----------
        *args
            Positional arguments for Axes.
        use_gravity_formatter : bool, optional
            Whether to use the gravity formatter. Default is False.
        **kwargs
            Keyword arguments for Axes.
        """
        # By default, use log-log scale
        xscale = kwargs.pop("xscale", "log")
        yscale = kwargs.pop("yscale", "log")
        use_gravity_formatter = kwargs.pop("use_gravity_formatter", False)

        super().__init__(*args, **kwargs)

        self.set_xscale(xscale)
        self.set_yscale(yscale)
        self.set_xlabel("Frequency (Hz)")
        self.set_ylabel("Velocity (m/s)")

        # Integration axis (displacement)
        self.iaxis = OrderAxis(self, 1)
        self.iaxis.set_major_formatter(DisplacementFormatter())

        # Derivation axis (acceleration)
        self.daxis = OrderAxis(self, -1)
        if use_gravity_formatter:
            self.daxis.set_major_locator(GravityLogLocator())
            self.daxis.set_major_formatter(GravityLogFormatter())
        else:
            self.daxis.set_major_formatter(AccelLogFormatter())

        if rcParams["axes.grid"]:
            self.iaxis.grid(True)
            self.daxis.grid(True)

    def grid(
        self, visible: bool | None = None, which: str = "major", axis: str = "both", **kwargs
    ) -> None:
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None, optional
            Whether to show the grid lines.
        which : {"major", "minor", "both"}, optional
            The grid lines to apply changes to.
        axis : {"both", "x", "y"}, optional
            The axis to apply changes to.
        **kwargs
            Style properties.
        """
        super().grid(visible, which=which, axis=axis, **kwargs)

        if visible is None:
            visible = rcParams["axes.grid"]

        if self.iaxis is not None:
            self.iaxis.grid(visible=visible, which=which, **kwargs)
        if self.daxis is not None:
            self.daxis.grid(visible=visible, which=which, **kwargs)

    def draw(self, renderer: RendererBase) -> None:
        """
        Draw the axes and its children.

        Parameters
        ----------
        renderer : matplotlib.backend_bases.RendererBase
            The renderer.
        """
        super().draw(renderer)
        self.iaxis.draw(renderer)
        self.daxis.draw(renderer)

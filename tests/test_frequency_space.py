import os
import sys

import matplotlib.pyplot as plt
import pytest

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.axes import OrderAxis
from vibro_nomogram.formatting import GravityLogFormatter, GravityLogLocator
from vibro_nomogram.frequency_space import FrequencySpaceAxes


class TestFrequencySpaceAxes:
    @pytest.fixture
    def ax(self):
        """Fixture to create a figure and FrequencySpaceAxes."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="frequency_space")
        yield ax
        plt.close(fig)

    def test_init(self, ax):
        """Test initialization of FrequencySpaceAxes."""
        assert isinstance(ax, FrequencySpaceAxes)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert ax.get_xlabel() == "Frequency (Hz)"
        assert ax.get_ylabel() == "Velocity (m/s)"

        assert isinstance(ax.iaxis, OrderAxis)
        assert ax.iaxis.order == 1
        assert isinstance(ax.daxis, OrderAxis)
        assert ax.daxis.order == -1

    def test_grid_delegation(self, ax):
        """Test that grid() calls are propagated to OrderAxes."""
        # Force disable first to be sure
        ax.grid(False)
        assert not ax.iaxis._grid_info["major"]["visible"]
        assert not ax.daxis._grid_info["major"]["visible"]

        # Enable grid
        ax.grid(True, which="major")

        assert ax.iaxis._grid_info["major"]["visible"]
        assert ax.daxis._grid_info["major"]["visible"]

        # Disable grid
        ax.grid(False)
        assert not ax.iaxis._grid_info["major"]["visible"]
        assert not ax.daxis._grid_info["major"]["visible"]

    def test_custom_formatters(self):
        """Test initialization with custom formatters flag."""
        fig = plt.figure()
        ax = fig.add_subplot(
            111, projection="frequency_space", use_gravity_formatter=True
        )

        assert isinstance(ax.daxis.get_major_formatter(), GravityLogFormatter)
        assert isinstance(ax.daxis.get_major_locator(), GravityLogLocator)

        plt.close(fig)

    def test_projection_registration(self):
        """Test that the projection is registered with Matplotlib."""
        from matplotlib.projections import get_projection_names

        assert "frequency_space" in get_projection_names()

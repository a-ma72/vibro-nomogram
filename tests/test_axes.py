import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.axes import OrderAxis
from vibro_nomogram.transforms import SpecTransform


class TestOrderAxis:
    @pytest.fixture
    def ax(self):
        """Fixture to create a figure and axes."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        yield ax
        plt.close(fig)

    def test_init(self, ax):
        """Test initialization of OrderAxis."""
        order = 1
        order_axis = OrderAxis(ax, order=order)

        assert order_axis.order == order
        assert isinstance(order_axis.transform, SpecTransform)
        assert order_axis.transform.order == order
        # transData should be composite
        assert order_axis.transData is not None

    def test_defaults(self, ax):
        """Test that default locators and formatters are set."""
        order_axis = OrderAxis(ax, order=1)

        # Check locators are LogLocators as set in _set_defaults
        assert order_axis.get_major_locator() is not None
        assert order_axis.get_minor_locator() is not None
        assert order_axis.get_major_formatter() is not None

        # Check grid info structure
        assert "major" in order_axis._grid_info
        assert "minor" in order_axis._grid_info
        assert not order_axis._grid_info["major"]["visible"]

    def test_get_ylim_displacement(self, ax):
        """Test get_ylim for displacement (order 1)."""
        order_axis = OrderAxis(ax, order=1)

        # Set limits: f=[1, 10], v=[1, 10]
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)

        # s = v / (2*pi*f)
        # Corners:
        # f=1, v=1 => s = 1/(2pi)
        # f=1, v=10 => s = 10/(2pi)
        # f=10, v=1 => s = 1/(20pi)
        # f=10, v=10 => s = 10/(20pi) = 1/(2pi)

        # min s = 1/(20pi) approx 0.0159
        # max s = 10/(2pi) approx 1.59

        ymin, ymax = order_axis.get_ylim()

        expected_min = 1 / (2 * np.pi * 10)
        expected_max = 10 / (2 * np.pi * 1)

        np.testing.assert_allclose(ymin, expected_min)
        np.testing.assert_allclose(ymax, expected_max)

    def test_get_ylim_acceleration(self, ax):
        """Test get_ylim for acceleration (order -1)."""
        order_axis = OrderAxis(ax, order=-1)

        # Set limits: f=[1, 10], v=[1, 10]
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)

        # a = v * (2*pi*f)
        # min a at min f, min v => 1 * 2pi
        # max a at max f, max v => 10 * 20pi = 200pi

        ymin, ymax = order_axis.get_ylim()

        expected_min = 1 * (2 * np.pi * 1)
        expected_max = 10 * (2 * np.pi * 10)

        np.testing.assert_allclose(ymin, expected_min)
        np.testing.assert_allclose(ymax, expected_max)

    def test_clip_lines_to_box(self, ax):
        """Test the line clipping logic."""
        order_axis = OrderAxis(ax, order=1)

        # Box: [0, 10] x [0, 10]
        x_min, x_max = 0.0, 10.0
        y_min, y_max = 0.0, 10.0
        m = 1.0

        # Case 1: Line y = x (c=0) -> (0,0) to (10,10)
        # Case 2: Line y = x + 5 (c=5) -> (0,5) to (5,10)
        # Case 3: Line y = x + 20 (c=20) -> Outside

        c = np.array([0.0, 5.0, 20.0])

        p1s, p2s, valid = order_axis._clip_lines_to_box(
            m, c, x_min, x_max, y_min, y_max
        )

        assert len(valid) == 3
        assert valid[0]
        assert valid[1]
        assert not valid[2]

        # Check coordinates for valid lines
        np.testing.assert_allclose(p1s[0], [0, 0])
        np.testing.assert_allclose(p2s[0], [10, 10])

        np.testing.assert_allclose(p1s[1], [0, 5])
        np.testing.assert_allclose(p2s[1], [5, 10])

    def test_grid_configuration(self, ax):
        """Test enabling/disabling grid."""
        order_axis = OrderAxis(ax, order=1)

        # Default is invisible
        assert not order_axis._grid_info["major"]["visible"]

        # Enable major grid
        order_axis.grid(True, which="major", color="red")
        assert order_axis._grid_info["major"]["visible"]
        assert order_axis._grid_info["major"]["style"]["color"] == "red"

        # Enable minor grid
        order_axis.grid(True, which="minor", linestyle="-")
        assert order_axis._grid_info["minor"]["visible"]
        assert order_axis._grid_info["minor"]["style"]["linestyle"] == "-"

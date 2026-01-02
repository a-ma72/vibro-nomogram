import os
import sys

import numpy as np

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.formatting import (
    AccelLogFormatter,
    DisplacementFormatter,
    GravityLogFormatter,
    GravityLogLocator,
)


class TestDisplacementFormatter:
    def test_zero(self):
        fmt = DisplacementFormatter()
        assert fmt(0) == "0"

    def test_micrometers(self):
        fmt = DisplacementFormatter()
        # 1e-6 m = 1 µm
        assert fmt(1e-6) == "1 µm"
        # 5.5e-5 m = 55 µm
        assert fmt(5.5e-5) == "55 µm"
        # 9.99e-4 m = 999 µm
        assert fmt(9.99e-4) == "999 µm"

    def test_millimeters(self):
        fmt = DisplacementFormatter()
        # 1e-3 m = 1 mm
        assert fmt(1e-3) == "1 mm"
        # 0.5 m = 500 mm
        assert fmt(0.5) == "500 mm"
        # 0.999 m = 999 mm
        assert fmt(0.999) == "999 mm"

    def test_meters(self):
        fmt = DisplacementFormatter()
        # 1.0 m
        assert fmt(1.0) == "1 m"
        # 100 m
        assert fmt(100) == "100 m"

    def test_negative(self):
        fmt = DisplacementFormatter()
        assert fmt(-1e-6) == "-1 µm"
        assert fmt(-0.5) == "-500 mm"
        assert fmt(-2.0) == "-2 m"


class TestAccelLogFormatter:
    def test_zero(self):
        fmt = AccelLogFormatter()
        assert fmt(0) == "0"
        assert fmt(1e-16) == "0"

    def test_integers(self):
        fmt = AccelLogFormatter()
        assert fmt(10.0) == "10 m/s²"
        # Close to integer
        assert fmt(10.0000000001) == "10 m/s²"

    def test_floats(self):
        fmt = AccelLogFormatter()
        assert fmt(9.81) == "9.81 m/s²"
        assert fmt(12345) == "1.234e+04 m/s²"


class TestGravityLogFormatter:
    def test_zero(self):
        fmt = GravityLogFormatter()
        assert fmt(0) == "0"

    def test_g_units(self):
        fmt = GravityLogFormatter()
        g = 9.81
        assert fmt(g) == "1 g"
        assert fmt(2 * g) == "2 g"
        assert fmt(0.5 * g) == "0.5 g"
        assert fmt(10 * g) == "10 g"

    def test_rounding(self):
        fmt = GravityLogFormatter()
        g = 9.81
        # Close to 1g
        assert fmt(g + 1e-10) == "1 g"


class TestGravityLogLocator:
    def test_tick_values(self):
        locator = GravityLogLocator()
        g = 9.81
        # Range covering 0.9g to 10.1g
        # LogLocator base 10 should find 10^0 (1) and 10^1 (10) in [0.9, 10.1]
        # So we expect ticks at 1g and 10g
        vmin = 0.9 * g
        vmax = 10.1 * g

        ticks = locator.tick_values(vmin, vmax)

        # We expect 1g and 10g to be in the ticks
        assert np.any(np.isclose(ticks, 1.0 * g))
        assert np.any(np.isclose(ticks, 10.0 * g))

    def test_tick_values_negative_range(self):
        # Should fallback to super behavior (likely empty or standard log ticks if handled)
        # This test primarily ensures it doesn't crash on non-positive limits
        locator = GravityLogLocator()
        ticks = locator.tick_values(-10, 10)
        assert isinstance(ticks, np.ndarray)

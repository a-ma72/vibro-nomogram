"""
Simple example demonstrating the FrequencySpaceAxes projection.

This script shows how to:
1. Initialize a plot with the 'frequency_space' projection.
2. Plot velocity data against frequency.
3. Add a constant displacement line on the integration axis (iaxis).
4. Customize the formatters for the displacement and acceleration axes.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# This import is important to register the projection
from vibro_nomogram import DisplacementFormatter

_twopi = 2.0 * np.pi


def main():
    fig, ax = plt.subplots(
        subplot_kw={
            "projection": "frequency_space",
            # "xscale": "log",
            # "yscale": "log",
        },
        figsize=(8, 6),
    )
    ax.iaxis.set_major_formatter(DisplacementFormatter())
    ax.daxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.2g} m/s²"))

    # Calculate velocity for constant displacement of 1 mm
    freq = np.logspace(0, 3, 100)
    # Velocity = Displacement * (2 * pi * f)  // Differentiation, order = -1
    velo = 1e-3 * (_twopi * freq)
    # Draw on velocity (main) axis
    ax.plot(freq, velo, "k-", label=r"Displacement $s = \frac{\nu}{\omega}$ in [m]")
    # Draw in integration axis (displacement), must be identical
    ax.iaxis.plot(
        freq,
        np.full_like(freq, 1e-3),
        "--",
        color="tab:red",
        label="Displacement $s$ in [m]",
    )

    ax.grid(True, which="both", linestyle="--", color="gray")
    ax.legend(loc=8)

    plt.show()


if __name__ == "__main__":
    main()

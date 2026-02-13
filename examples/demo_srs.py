"""
Demonstrates plotting a Shock Response Spectrum (SRS) using the FrequencySpaceAxes.

This example shows how to:
1. Define an SRS profile with breakpoints in frequency and acceleration (g).
2. Initialize the plot with `use_gravity_formatter=True` to display acceleration in 'g'.
3. Plot the profile on the acceleration axis (`daxis`).
4. Add exclusion zones (filled regions) for displacement, velocity, and acceleration limits.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.axes import OrderAxis


def get_srs_points():
    """
    Returns frequency and velocity arrays for a typical SRS profile.
    Profile:
      - 10 Hz to 100 Hz: Constant Velocity (ramp up in acceleration)
      - 100 Hz to 2000 Hz: Constant Acceleration (plateau)

    Breakpoints:
      - 10 Hz, 10 g
      - 100 Hz, 100 g
      - 2000 Hz, 100 g
    """
    g = 9.81  # m/s²
    # Breakpoints in Frequency (Hz) and Acceleration (g)
    f_breakpoints = np.array([10.0, 100.0, 2000.0])
    a_breakpoints_g = np.array([10.0, 100.0, 100.0])

    return f_breakpoints, a_breakpoints_g * g


def main():
    freqs, accels = get_srs_points()

    fig = plt.figure(figsize=(10, 8))
    # use_gravity_formatter=True enables 'g' units on the acceleration axis
    ax: OrderAxis = fig.add_subplot(
        111, projection="frequency_space", use_gravity_formatter=True
    )

    ax.daxis.plot(freqs, accels, "r-", linewidth=2, label="SRS Profile")
    # Recompute the data limits based on current artists
    ax.relim()
    # Autoscale the view limits using the data limits
    ax.autoscale_view()
    # Now freeze both axes
    ax.set_autoscale_on(False)
    ax.set_ylim(0.01, 10)

    _, ymax = ax.get_ylim()
    ax.fill_between(
        np.logspace(*np.log10(ax.get_xlim()), 50),
        3,
        ymax,  # m/s
        color="tab:red",
        alpha=0.3,
    )
    _, ymax = ax.daxis.get_ylim()
    ax.fill_between(
        np.logspace(*np.log10(ax.get_xlim()), 50),
        3000,
        ymax,  # m/s²
        color="tab:red",
        transform=ax.daxis.transData,
        alpha=0.3,
    )
    _, ymax = ax.iaxis.get_ylim()
    ax.fill_between(
        np.logspace(*np.log10(ax.get_xlim()), 50),
        0.03,
        ymax,  # m
        color="tab:red",
        transform=ax.iaxis.transData,
        alpha=0.3,
    )

    ax.grid(visible=True, which="major", color="k", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.grid(visible=True, which="minor", color="k", linewidth=0.5, linestyle="-", alpha=0.2)

    ax.legend(loc=8)
    ax.set_title("Vibro-Nomogram SRS Demonstrator")

    plt.show()


if __name__ == "__main__":
    main()

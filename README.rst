Matplotlib Double-Log Projection with Acceleration and Displacement Axes
========================================================================

.. figure:: https://raw.githubusercontent.com/a-ma72/vibro-nomogram/main/demo.png



This project provides a custom projection for Matplotlib to create double-logarithmic plots of velocity vs. frequency,
with additional-tilted axes for acceleration and displacement.

Features
--------

- Log-log plot for velocity vs. frequency.
- Additional axes for acceleration and displacement.
- Customizable grid lines.

Installation
------------

To install this package, you can run:

.. code-block:: bash

    pip install .

from the root of this repository.

Usage
-----

Here is a simple example of how to use this projection:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import vibro_nomogram

    def main():
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='frequency_space'))

        # plot some data
        f = np.logspace(-1, 3, 100)
        v = 100 * np.exp(-f/1000)

        ax.plot(f, v, 'k-')
        ax.grid(True, which='both')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Velocity (mm/s)')
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(1e-2, 1e3)

        plt.show()

    if __name__ == '__main__':
        main()
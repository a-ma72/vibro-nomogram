import os
import sys
import unittest

import matplotlib.pyplot as plt
from matplotlib.projections import register_projection

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.frequency_space import FrequencySpaceAxes


class TestProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_projection(FrequencySpaceAxes)

    def test_init(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="frequency_space")
        self.assertIsInstance(ax, FrequencySpaceAxes)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

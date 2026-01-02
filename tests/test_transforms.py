import os
import sys

import numpy as np

try:
    import vibro_nomogram
except ImportError:
    # Ensure local package is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vibro_nomogram.transforms import SpecTransform, _eps


class TestSpecTransform:
    def test_init(self):
        """Test initialization of SpecTransform."""
        transform = SpecTransform(order=1)
        assert transform.order == 1
        assert transform.is_inverse is False
        assert transform.input_dims == 2
        assert transform.output_dims == 2

        transform_inv = SpecTransform(order=2, is_inverse=True)
        assert transform_inv.order == 2
        assert transform_inv.is_inverse is True

    def test_transform_identity(self):
        """Test transformation with order 0 (identity)."""
        transform = SpecTransform(order=0)
        data = np.array([[1.0, 5.0], [10.0, 20.0]])
        result = transform.transform(data)

        # Should be identical to input
        np.testing.assert_allclose(result, data)

    def test_transform_integration(self):
        """Test transformation with order 1 (integration: y / w)."""
        transform = SpecTransform(order=1)
        # f = 1/(2pi) => w = 1. y = 10. Result y = 10/1 = 10.
        f = 1.0 / (2 * np.pi)
        data = np.array([[f, 10.0]])
        result = transform.transform(data)

        expected = np.array([[f, 10.0]])
        np.testing.assert_allclose(result, expected)

        # f = 1 => w = 2pi. y = 2pi. Result y = 1.
        data2 = np.array([[1.0, 2 * np.pi]])
        result2 = transform.transform(data2)
        expected2 = np.array([[1.0, 1.0]])
        np.testing.assert_allclose(result2, expected2)

    def test_transform_differentiation(self):
        """Test transformation with order -1 (differentiation: y * w)."""
        transform = SpecTransform(order=-1)
        # f = 1 => w = 2pi. y = 1. Result y = 1 * 2pi.
        data = np.array([[1.0, 1.0]])
        result = transform.transform(data)

        expected = np.array([[1.0, 2 * np.pi]])
        np.testing.assert_allclose(result, expected)

    def test_inverted(self):
        """Test the inverted method returns correct inverse transform."""
        transform = SpecTransform(order=1)
        inverse = transform.inverted()

        assert isinstance(inverse, SpecTransform)
        assert inverse.order == 1
        assert inverse.is_inverse is True

        # Double inversion should return to original state (logic-wise)
        inverse_inverse = inverse.inverted()
        assert inverse_inverse.is_inverse is False

    def test_round_trip(self):
        """Test that transform -> inverse transform returns original data."""
        transform = SpecTransform(order=2)
        data = np.array([[1.0, 10.0], [100.0, 0.01]])

        forward = transform.transform(data)
        backward = transform.inverted().transform(forward)

        np.testing.assert_allclose(backward, data)

    def test_clipping(self):
        """Test that values <= 0 are clipped to epsilon."""
        transform = SpecTransform(order=0)
        # Input with zeros and negative numbers
        data = np.array([[0.0, 0.0], [-5.0, -5.0]])
        result = transform.transform(data)

        # Check that output values are positive (>= epsilon)
        assert np.all(result > 0)
        # Specifically, they should be clipped to _eps defined in transforms.py
        # Since order is 0, y_out = y_in_clipped
        # x_out = x_in_clipped
        expected_val = _eps
        np.testing.assert_allclose(result, expected_val)

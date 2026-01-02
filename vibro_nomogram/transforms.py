from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib.transforms import Transform

_eps = 1e-9
_twopi = 2.0 * np.pi
_log_twopi = np.log10(_twopi)


class SpecTransform(Transform):
    """
    Transform for spectral data integration/differentiation.

    This transform handles the conversion between physical quantities related by
    differentiation or integration in the frequency domain (e.g., displacement,
    velocity, acceleration).

    Parameters
    ----------
    order : int, optional
        The order of differentiation/integration.
        -1 for differentiation (e.g., velocity to acceleration).
        +1 for integration (e.g., velocity to displacement).
        Default is 0.
    is_inverse : bool, optional
        If True, applies the inverse transformation. Default is False.
    """

    input_dims = 2
    output_dims = 2
    is_inverse: bool

    # Order of differentiation / integration
    # Examples:
    #   -1 -> first derivative (s -> v -> a)
    #   -2 -> second derivative (s -> a)
    #   +1 -> first integral (s -> v -> a)
    #   +2 -> second (iterated) integral (s -> a)
    #    0 -> identity (no operation)
    order: int = 0

    @staticmethod
    def _forward(f, y, order):
        """Forward transform: Y = y / (2 * pi * f)^order"""
        w = _twopi * f
        return y / w**order

    @staticmethod
    def _log_forward(f, y, order):
        """Logarithmic forward transform: y_primYe = y / (2 * pi * f)^order"""
        log_w = _log_twopi + np.log10(f)
        return np.log10(y) - order * log_w

    @staticmethod
    def _inverse(f, y, order):
        """Inverse transform: y = Y * (2 * pi * f)^order"""
        w = _twopi * f
        return y * w**order

    @staticmethod
    def _log_inverse(f, y, order):
        """Logarithmic inverse transform: y = Y * (2 * pi * f)^order"""
        log_w = _log_twopi + np.log10(f)
        return np.log10(y) + order * log_w

    def __init__(self, order: int = 0, is_inverse: bool = False) -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        order : int, optional
            Order of differentiation/integration. Default is 0.
        is_inverse : bool, optional
            Whether this is the inverse transform. Default is False.
        """
        super().__init__(shorthand_name=self.__class__.__name__)
        self.order = order
        self.is_inverse = is_inverse

    def transform_non_affine(
        self, values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Apply the transform to the given values.

        Parameters
        ----------
        values : array_like
            Input coordinates (N, 2).

        Returns
        -------
        ndarray
            Transformed coordinates (N, 2).
        """
        # values is an (N, 2) numpy array of coordinates
        shaped_values = np.asarray(values)
        f = np.clip(shaped_values[:, 0], _eps, None)
        mag = np.clip(shaped_values[:, 1], _eps, None)

        # Initialize output array
        out = np.empty_like(shaped_values)

        # X component stays as is
        out[:, 0] = f

        if self.is_inverse:
            out[:, 1] = SpecTransform._inverse(f, mag, self.order)
        else:
            out[:, 1] = SpecTransform._forward(f, mag, self.order)

        return np.reshape(out, values.shape)

    def inverted(self) -> SpecTransform:
        """
        Return the inverse transform.

        Returns
        -------
        SpecTransform
            The inverted transform.
        """
        return SpecTransform(self.order, not self.is_inverse)

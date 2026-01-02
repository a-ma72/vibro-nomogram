import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt


class DisplacementFormatter(mticker.Formatter):
    """
    Formatter for displacement values, automatically scaling to m, mm, or µm.

    Formats values with 5 significant digits.
    """

    def __call__(self, x, pos=None) -> str:
        if x == 0:
            return "0"

        abs_x = abs(x)
        if abs_x < 1e-3:
            # Convert to µm
            val = x * 1e6
            unit = "µm"
        elif abs_x < 1.0:
            # Convert to mm
            val = x * 1e3
            unit = "mm"
        else:
            val = x
            unit = "m"

        return f"{val:.5g} {unit}"


class GravityLogLocator(mticker.LogLocator):
    """
    Locator for ticks at multiples of g (9.81 m/s^2) in log scale.
    """

    def __init__(self, base: float = 10.0, subs=(1.0,), numticks: int = 15) -> None:
        super().__init__(base=base, subs=subs, numticks=numticks)
        self._g = 9.81

    def tick_values(self, vmin: float, vmax: float) -> npt.NDArray[np.float64]:
        if vmin <= 0 or vmax <= 0:
            return np.array([])

        vmin_g = vmin / self._g
        vmax_g = vmax / self._g

        ticks_g = super().tick_values(vmin_g, vmax_g)

        return ticks_g * self._g


class AccelLogFormatter(mticker.Formatter):
    """
    Formatter for acceleration values in m/s^2.
    """

    def __call__(self, x, pos: int = None) -> str:
        if abs(x) < 1e-15:
            return "0"

        if abs(x - round(x)) < 1e-9:
            x = round(x)

        s = f"{x:.4g}"

        return f"{s} m/s²"


class GravityLogFormatter(mticker.Formatter):
    """
    Formatter for acceleration values in g (9.81 m/s^2).
    """

    def __call__(self, x, pos: int = None) -> str:
        g = 9.81
        val_g = x / g

        if abs(val_g) < 1e-15:
            return "0"

        if abs(val_g - round(val_g)) < 1e-9:
            val_g = round(val_g)

        s = f"{val_g:.4g}"

        return f"{s} g"

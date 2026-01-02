from matplotlib.projections import register_projection

from .formatting import (
    AccelLogFormatter,
    DisplacementFormatter,
    GravityLogFormatter,
    GravityLogLocator,
)
from .frequency_space import FrequencySpaceAxes

__all__ = [
    "FrequencySpaceAxes",
    "AccelLogFormatter",
    "DisplacementFormatter",
    "GravityLogFormatter",
    "GravityLogLocator",
]

register_projection(FrequencySpaceAxes)

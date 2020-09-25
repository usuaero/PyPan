"""Defines classes for solving potential flow scenarios."""

import numpy as np


class VortexRingSolver:
    """Vortex ring solver."""

    def __init__(self, **kwargs):

        # Store param dict
        self._input_dict = kwargs["param_dict"]
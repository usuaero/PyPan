"""Defines classes for solving potential flow scenarios."""

import numpy as np


class VortexRingSolver:
    """Vortex ring solver.

    Parameters
    ----------
    mesh : Mesh
        A mesh object.

    atmosphere : dict
        A dictionary of atmospherics properties.
    
    """

    def __init__(self, **kwargs):

        # Store mesh
        self._mesh = kwargs["mesh"]
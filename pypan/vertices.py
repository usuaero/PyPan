"""Defines classes for handling vertices (necessary for linear singularity distributions)."""

import copy

import numpy as np

from pypan.pp_math import norm


class Vertex:
    """A class defining a mesh vertex. Stores information regarding singularity strengths, domains of dependence, etc.

    Parameters
    ----------
    r : list
        Coordinates of vertex.

    N_vert : int
        Total number of vertices in the mesh this vertex belongs to.
    """

    def __init__(self, r, N_vert):

        # Store vertex
        self.r = copy.deepcopy(r)
        self.phi = 0.0
        self.dod_list = []
        self.dod_array = np.zeros(N_vert, dtype=bool)
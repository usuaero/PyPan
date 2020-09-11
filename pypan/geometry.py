"""Defines classes for handling geometric panels and meshes."""

import stl

import numpy as np


class Panel:
    """A base class defining a panel for potential flow simulation."""
    pass


class Quad(Panel):
    """A quadrilateral panel."""

    def __init__(self, v0, v1, v2, v3, n=None):
        super().__init__()


class Tri(Panel):
    """A triangular panel."""

    def __init__(self, v0, v1, v2, n=None):
        super().__init__()


class Mesh:
    """A class for defining collections of panels."""

    def __init__(self, **kwargs):

        # Load mesh
        self._load_mesh()

        # Check mesh
        self._check_mesh()

    
    def _load_mesh(self):
        # Loads the mesh from the input file
        pass


    def _check_mesh(self):
        # Checks the mesh is appropriate
        pass
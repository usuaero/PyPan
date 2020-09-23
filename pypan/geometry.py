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

        # Store vectors
        self.v0 = v0
        self.v1 = v2
        self.v2 = v2
        print(self.v0)
        print(self.v1)
        print(self.v2)
        self.n = n

        # Check normal
        if self.n is None:
            d1 = self.v1-self.v0
            d2 = self.v2-self.v1
            N = np.cross(d1, d2)
            self.n = N/np.linalg.norm(N)
        else:
            self.n = self.n/np.linalg.norm(self.n)

        # Calculate area
        nx, ny, nz = self.n
        x0, y0, z0 = self.v0
        x1, y1, z1 = self.v1
        x2, y2, z2 = self.v2
        dA0 = 0.5*ny*(z1+z0)*(x1-x0)\
             +0.5*nz*(x1+x0)*(y1-y0)\
             +0.5*nx*(y1+y0)*(z1-z0)
        dA1 = 0.5*ny*(z2+z1)*(x2-x1)\
             +0.5*nz*(x2+x1)*(y2-y1)\
             +0.5*nx*(y2+y1)*(z2-z1)
        dA2 = 0.5*ny*(z0+z2)*(x0-x2)\
             +0.5*nz*(x0+x2)*(y0-y2)\
             +0.5*nx*(y0+y2)*(z0-z2)
        self.A = dA0+dA1+dA2
        print(self.A)

        # Calculate centroid


class Mesh:
    """A class for defining collections of panels."""

    def __init__(self, **kwargs):

        # Load kwargs
        mesh_file = kwargs.get("mesh_file")
        mesh_type = kwargs.get("mesh_file_type")
        self._verbose = kwargs.get("verbose", False)

        # Load mesh
        self._load_mesh(mesh_file, mesh_type)

        # Check mesh
        self._check_mesh()

    
    def _load_mesh(self, mesh_file, mesh_file_type):
        # Loads the mesh from the input file

        # STL
        if mesh_file_type == "STL":
            self._load_stl(mesh_file)
        else:
            raise IOError("{0} is not a supported mesh type for PyPan.")


    def _load_stl(self, stl_file):
        # Loads mesh from an stl file

        # Load stl file
        self._raw_stl_mesh = stl.mesh.Mesh.from_file(stl_file)

        # Initialize panel objects
        self.N = self._raw_stl_mesh.v0.shape[0]
        if self._verbose: print("Successfully read STL file with {0} facets.".format(self.N))
        self.panels = np.zeros(self.N, dtype=Tri)
        for i in range(self.N):
            self.panels[i] = Tri(self._raw_stl_mesh.v0[i],
                                 self._raw_stl_mesh.v1[i],
                                 self._raw_stl_mesh.v2[i],
                                 n=self._raw_stl_mesh.normals[i])

        if self._verbose: print("{0} mesh panels successfully initialized.".format(self.N))


    def _check_mesh(self):
        # Checks the mesh is appropriate and determines where Kutta condition should exist
        pass
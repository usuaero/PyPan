"""Defines classes for handling geometric panels and meshes."""

import stl
import warnings

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


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
        self.v1 = v1
        self.v2 = v2
        self.n = n

        # Get normal vector
        if self.n is None:
            self._calc_normal()
        else:
            try:
                self.n = self.n/np.linalg.norm(self.n)
            except RuntimeWarning:
                self._calc_normal()

        # Determine area
        self._calc_area()

        # Determine centroid
        self._calc_centroid()


    def _calc_area(self):
        # Calculates the panel area

        # Get vector components
        nx, ny, nz = self.n
        x0, y0, z0 = self.v0
        x1, y1, z1 = self.v1
        x2, y2, z2 = self.v2

        # Get the area using Stoke's theorem
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


    def _calc_normal(self):
        # Calculates the panel unit normal vector
        d1 = self.v1-self.v0
        d2 = self.v2-self.v1
        N = np.cross(d1, d2)
        self.n = N/np.linalg.norm(N)

    
    def _calc_centroid(self):
        # Calculates the location of the panel centroid

        # Get primed coordinate system
        x_p = self.v1-self.v0
        x_p = x_p/np.linalg.norm(x_p)
        z_p = np.copy(self.n)
        y_p = np.cross(z_p, x_p)

        # Construct transformation matrix
        T = np.concatenate((x_p[:,np.newaxis], y_p[:,np.newaxis], z_p[:,np.newaxis]), axis=1).T
        
        # Transform vertices
        v0_p = np.matmul(T, self.v0[:,np.newaxis]).flatten()
        v1_p = np.matmul(T, self.v1[:,np.newaxis]).flatten()
        v2_p = np.matmul(T, self.v2[:,np.newaxis]).flatten()

        # Get transformed coordinates of centroid
        x_c_p = 1.0/(6.0*self.A)*(
            (v0_p[0]+v1_p[0])*(v0_p[0]*v1_p[1]-v1_p[0]*v0_p[1])+
            (v1_p[0]+v2_p[0])*(v1_p[0]*v2_p[1]-v2_p[0]*v1_p[1])+
            (v2_p[0]+v0_p[0])*(v2_p[0]*v0_p[1]-v0_p[0]*v2_p[1]))
        y_c_p = 1.0/(6.0*self.A)*(
            (v0_p[1]+v1_p[1])*(v0_p[0]*v1_p[1]-v1_p[0]*v0_p[1])+
            (v1_p[1]+v2_p[1])*(v1_p[0]*v2_p[1]-v2_p[0]*v1_p[1])+
            (v2_p[1]+v0_p[1])*(v2_p[0]*v0_p[1]-v0_p[0]*v2_p[1]))

        # Transform back to standard coordinates
        v_cg_p = np.array([x_c_p, y_c_p, 0.0])[:,np.newaxis]
        self.v_cg = np.matmul(T.T, v_cg_p).flatten()


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
        self.panels = np.empty(self.N, dtype=Tri)
        for i in range(self.N):
            self.panels[i] = Tri(self._raw_stl_mesh.v0[i],
                                 self._raw_stl_mesh.v1[i],
                                 self._raw_stl_mesh.v2[i],
                                 n=self._raw_stl_mesh.normals[i])

            # Check for zero area
            if abs(self.panels[i].A)<1e-10:
                raise IOError("Panel {0} in the mesh has zero area.".format(i))

        if self._verbose: print("{0} mesh panels successfully initialized.".format(self.N))


    def _check_mesh(self):
        # Checks the mesh is appropriate and determines where Kutta condition should exist
        pass


    def plot(self, **kwargs):
        """Plots the mesh in 3D.

        Parameters
        ----------
        vertices : bool, optional
            Whether to display vertices. Defaults to True.

        centroids : bool, optional
            Whether to display centroids. Defaults to True.
        """

        # Set up plot
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        if kwargs.get("vertices", True):
            for i in range(self.N):
                panel = self.panels[i]
                ax.plot(panel.v0[0], panel.v0[1], panel.v0[2], 'b.')
                ax.plot(panel.v1[0], panel.v1[1], panel.v1[2], 'b.')
                ax.plot(panel.v2[0], panel.v2[1], panel.v2[2], 'b.')
        
        # Plot centroids
        if kwargs.get("centroids", True):
            for i in range(self.N):
                panel = self.panels[i]
                ax.plot(panel.v_cg[0], panel.v_cg[1], panel.v_cg[2], 'r.')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(5.0, -5.0)
        ax.set_ylim3d(-5.0, 5.0)
        ax.set_zlim3d(5.0, -5.0)
        plt.show()
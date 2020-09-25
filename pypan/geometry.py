"""Defines classes for handling geometric panels and meshes."""

import stl
import warnings

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class Panel:
    """A base class defining a panel for potential flow simulation."""

    def __init__(self):
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

        # Construct transformation matrix
        T = np.zeros((3,3))
        T[0,:] = self.v1-self.v0
        T[0,:] /= np.linalg.norm(T[0,:])
        T[1,:] = np.cross(self.n, T[0,:])
        T[2,:] = np.copy(self.n)
        
        # Transform vertices
        v0_p = np.einsum('ij,j', T, self.v0)
        v1_p = np.einsum('ij,j', T, self.v1)
        v2_p = np.einsum('ij,j', T, self.v2)

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
        self.v_cg = np.einsum('ji,j', T, np.array([x_c_p, y_c_p, v0_p[2]]))


    def get_plot_list(self):
        """Returns a list of the vertex coordinates for plotting."""
        return [[self.v0[0], self.v1[0], self.v2[0], self.v0[0]],
                [self.v0[1], self.v1[1], self.v1[1], self.v0[1]],
                [self.v0[2], self.v1[2], self.v2[2], self.v0[2]]]


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
        self._check_mesh(**kwargs)

    
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
        if self._verbose:
            print("Successfully read STL file with {0} facets.".format(self.N))
            print("Initializing mesh panels (vertices, normals, areas, centroids, etc.)...", end='')
        self.panels = np.empty(self.N, dtype=Tri)
        for i in range(self.N):
            self.panels[i] = Tri(self._raw_stl_mesh.v0[i],
                                 self._raw_stl_mesh.v1[i],
                                 self._raw_stl_mesh.v2[i],
                                 n=self._raw_stl_mesh.normals[i])

            # Check for zero area
            if abs(self.panels[i].A)<1e-10:
                raise IOError("Panel {0} in the mesh has zero area.".format(i))

        if self._verbose: print("Finished.")


    def _check_mesh(self, **kwargs):
        # Checks the mesh is appropriate and determines where Kutta condition should exist

        # Get Kutta angle
        theta_K = np.radians(kwargs.get("kutta_angle", None))

        # Look for adjacent panels where the Kutta condition should be applied
        if theta_K is not None:
            if self._verbose: print("Determining location of Kutta condition...", end='')

            # Store edges
            wake_edges = []

            # Loop through possible combinations
            for i in range(self.N):
                panel_i = self.panels[i]

                # Start at the (i+1)th panel, so we don't repeat ourselves
                for j in range(i+1, self.N):
                    panel_j = self.panels[j]
                    
                    # Determine panel angle first, because that's cheaper
                    with np.errstate(invalid='ignore'):
                        theta = abs(np.arccos(np.einsum('i,i', panel_i.n, panel_j.n)))

                    # Store if greater than the Kutta angle
                    if theta > theta_K:

                        # Determine if we're adjacent
                        edge = []
                        for vi in [panel_i.v0, panel_i.v1, panel_i.v2]:
                            for vj in [panel_j.v0, panel_j.v1, panel_j.v2]:

                                # Check distance
                                d = np.linalg.norm(vi-vj)
                                if d<1e-8:
                                    edge.append(vi)

                                # Check we have two vertices already
                                if len(edge) == 2:
                                    wake_edges.append(edge)
                                    break


            self.wake_edges = np.array(wake_edges)
            if self._verbose: print("Finished. {0} Kutta edges detected.".format(self.wake_edges.shape[0]))


    def plot(self, **kwargs):
        """Plots the mesh in 3D.

        Parameters
        ----------
        panels : bool, optional
            Whether to display panels. Defaults to True.

        centroids : bool, optional
            Whether to display centroids. Defaults to True.

        kutta_edges : bool, optional
            Whether to display the edges at which the Kutta condition will be enforced.
            Defaults to True.
        """

        # Set up plot
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        if kwargs.get("panels", True):
            for i in range(self.N):
                ax.plot(*self.panels[i].get_plot_list(),'k-')
        
        # Plot centroids
        if kwargs.get("centroids", True):
            for i in range(self.N):
                panel = self.panels[i]
                ax.plot(panel.v_cg[0], panel.v_cg[1], panel.v_cg[2], 'r.')

        # Plot Kutta edges
        if kwargs.get("kutta_edges", True) and hasattr(self, "wake_edges"):
            for edge in self.wake_edges:
                ax.plot(edge[:,0], edge[:,1], edge[:,2], 'b')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        lims= ax.get_ylim()
        ax.set_xlim3d(lims[0], lims[1])
        ax.set_zlim3d(lims[0], lims[1])
        plt.show()
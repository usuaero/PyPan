"""Defines classes for handling geometric panels and meshes."""

import stl
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

from .pp_math import vec_norm, norm, vec_inner, inner, vec_cross, cross
from .helpers import OneLineProgress


class Panel:
    """A base class defining a panel for potential flow simulation."""

    def __init__(self, **kwargs):

        # Get normal vector
        self.n = kwargs.get("n")
        if self.n is None:
            self._calc_normal()
        else:
            try:
                self.n = self.n/norm(self.n)
            except RuntimeWarning:
                self._calc_normal()

        # Determine area
        self.A = kwargs.get("A", None)
        if self.A is None:
            self._calc_area()

        # Determine centroid
        self.v_c = kwargs.get("v_c", None)
        if self.v_c is None:
            self._calc_centroid()

        # Determine max side length
        self.d_max = kwargs.get("d_max", None)
        if self.d_max is None:
            self.d_max = np.max(vec_norm(self.vertices-np.roll(self.vertices, 1, axis=0)))


    def _calc_normal(self):
        # Calculates the panel unit normal vector
        # Assumes the panel is planar
        d1 = self.vertices[1]-self.vertices[0]
        d2 = self.vertices[2]-self.vertices[1]
        N = cross(d1, d2)
        self.n = N/norm(N)


    @abstractmethod
    def _calc_centroid(self):
        pass


    @abstractmethod
    def _calc_area(self):
        pass


class Quad(Panel):
    """A quadrilateral panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((4,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")
        self.vertices[3] = kwargs.get("v3")

        super().__init__(**kwargs)


class Tri(Panel):
    """A triangular panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")

        super().__init__(**kwargs)


    def _calc_area(self):
        # Calculates the panel area

        # Get vector components
        nx, ny, nz = self.n
        x0, y0, z0 = self.vertices[0]
        x1, y1, z1 = self.vertices[1]
        x2, y2, z2 = self.vertices[2]

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

    
    def _calc_centroid(self):
        # Calculates the location of the panel centroid

        # Construct transformation matrix
        T = np.zeros((3,3))
        T[0] = self.vertices[1]-self.vertices[0]
        T[0] /= norm(T[0])
        T[1] = cross(self.n, T[0])
        T[2] = np.copy(self.n)
        
        # Transform vertices
        v0_p = np.einsum('ij,j', T, self.vertices[0])
        v1_p = np.einsum('ij,j', T, self.vertices[1])
        v2_p = np.einsum('ij,j', T, self.vertices[2])

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
        self.v_c = np.einsum('ji,j', T, np.array([x_c_p, y_c_p, v0_p[2]]))


    def __str__(self):
        s = "P "+" ".join(["{:<20}"]*17)
        s = s.format(self.vertices[0,0],
                     self.vertices[0,1],
                     self.vertices[0,2],
                     self.vertices[1,0],
                     self.vertices[1,1],
                     self.vertices[1,2],
                     self.vertices[2,0],
                     self.vertices[2,1],
                     self.vertices[2,2],
                     self.n[0],
                     self.n[1],
                     self.n[2],
                     self.v_c[0],
                     self.v_c[1],
                     self.v_c[2],
                     self.A,
                     self.d_max)
        return s


    def get_ring_influence(self, points):
        """Determines the velocity vector induced by this panel at arbitrary
        points, assuming a vortex ring (0th order) model and a unit positive
        vortex strength.

        Parameters
        ----------
        points : ndarray
            An array of points where the first index is the point index and 
            the second index is the coordinate.

        Returns
        -------
        ndarray
            The velocity vector induced at each point.
        """

        # Determine displacement vectors
        r0 = points-self.vertices[0,:]
        r1 = points-self.vertices[1,:]
        r2 = points-self.vertices[2,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)[:,np.newaxis]
        r1_mag = vec_norm(r1)[:,np.newaxis]
        r2_mag = vec_norm(r2)[:,np.newaxis]

        # Calculate influence
        v_01 = ((r0_mag+r1_mag)*vec_cross(r0, r1))/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1)[:,np.newaxis]))
        v_12 = ((r1_mag+r2_mag)*vec_cross(r1, r2))/(r1_mag*r2_mag*(r1_mag*r2_mag+vec_inner(r1, r2)[:,np.newaxis]))
        v_20 = ((r2_mag+r0_mag)*vec_cross(r2, r0))/(r2_mag*r0_mag*(r2_mag*r0_mag+vec_inner(r2, r0)[:,np.newaxis]))

        return 0.25/np.pi*(v_01+v_12+v_20)


class KuttaEdge:
    """A class for defining an edge segment at which the Kutta condition is applied.

    Parameters
    ----------
    v0 : ndarray
        Start vertex.

    v1 : ndarray
        End vertex.

    panel_indices : list
        Indices (within the mesh) of the panels neighboring this edge.
    """

    def __init__(self, v0, v1, panel_indices):

        # Store
        self.vertices = np.zeros((2, 3))
        self.vertices[0] = v0
        self.vertices[1] = v1
        self.panel_indices = panel_indices


    def __str__(self):
        s = "E "+" ".join(["{:<20}"]*6)+" {} {}"
        s = s.format(self.vertices[0,0],
                     self.vertices[0,1],
                     self.vertices[0,2],
                     self.vertices[1,0],
                     self.vertices[1,1],
                     self.vertices[1,2],
                     self.panel_indices[0],
                     self.panel_indices[1])
        return s


class Mesh:
    """A class for defining collections of panels.

    Parameters
    ----------
    mesh_file : str
        File path to the mesh file.

    mesh_file_type : str
        The type of mesh file being loaded. Can be "STL" or "pypan".

    kutta_angle : float, optional
        The angle threshold for determining where the Kutta condition should
        be enforced. Defaults to None (in which case, lifting bodies may not
        be analyzed, except for "pypan" type meshes). 

        This is not needed for "pypan" type meshes. However, if given, this 
        will force reevaluation of the Kutta edges (expensive!) whether or not
        were determined previously. If not given, this is skipped.
    
    """

    def __init__(self, **kwargs):

        # Load kwargs
        mesh_file = kwargs.get("mesh_file")
        mesh_type = kwargs.get("mesh_file_type")
        self._verbose = kwargs.get("verbose", False)

        # Load mesh
        self._load_mesh(mesh_file, mesh_type)

        # Check mesh
        if mesh_type != "pypan" or (mesh_type == "pypan" and kwargs.get("kutta_angle", None) is None):
            self._check_mesh(**kwargs)

    
    def _load_mesh(self, mesh_file, mesh_file_type):
        # Loads the mesh from the input file
        start_time = time.time()

        # STL
        if mesh_file_type == "STL":
            self._load_stl(mesh_file)

        # PyPan
        elif mesh_file_type == "pypan":
            self._load_pypan_mesh(mesh_file)

        # Unrecognized type
        else:
            raise IOError("{0} is not a supported mesh type for PyPan.")

        end_time = time.time()
        if self._verbose: print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)


    def _load_stl(self, stl_file):
        # Loads mesh from an stl file

        # Load stl file
        raw_mesh = stl.mesh.Mesh.from_file(stl_file)

        # Initialize storage
        self.N = raw_mesh.v0.shape[0]
        self.panels = np.empty(self.N, dtype=Tri)
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)

        if self._verbose:
            print("\nSuccessfully read STL file with {0} facets.".format(self.N))
            print("\nInitializing mesh panels (vertices, normals, areas, centroids, etc.)...", end='', flush=True)

        # Loop through panels and initialize objects
        for i in range(self.N):

            # Initialize
            panel = Tri(v0=raw_mesh.v0[i],
                        v1=raw_mesh.v1[i],
                        v2=raw_mesh.v2[i],
                        n=raw_mesh.normals[i])

            # Check for zero area
            if abs(panel.A)<1e-10:
                raise IOError("Panel {0} in the mesh has zero area.".format(i))

            # Store
            self.panels[i] = panel
            self.cp[i] = panel.v_c
            self.n[i] = panel.n
            self.dA[i] = panel.A

    
    def _load_pypan_mesh(self, pypan_file):
        # Loads mesh from PyPan file

        # Read in file
        with open(pypan_file, 'r') as mesh_file_handle:
            lines = mesh_file_handle.read().splitlines()

        # Determine number of panels and edges
        for i, line in enumerate(lines):
            if line.split()[0] == "E":
                self.N = i
                break
        N_edges = len(lines)-self.N

        if self._verbose:
            print("\nDetected {0} panels and {1} Kutta edges in mesh file.".format(self.N, N_edges))

        # Initialize storage
        self.panels = []
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)

        if self._verbose:
            print("\nParsing mesh panels (vertices, normals, areas, centroids, etc.)...", end='', flush=True)

        # Loop through panels and initialize objects
        for i in range(self.N):

            # Split line
            info = lines[i].split()

            # Initialize panel
            if len(info) == 18:
                panel = Tri(v0=np.array([float(info[1]), float(info[2]), float(info[3])]),
                            v1=np.array([float(info[4]), float(info[5]), float(info[6])]),
                            v2=np.array([float(info[7]), float(info[8]), float(info[9])]),
                            n=np.array([float(info[10]), float(info[11]), float(info[12])]),
                            v_c=np.array([float(info[13]), float(info[14]), float(info[15])]),
                            A=float(info[16]),
                            d_max=float(info[17]))
            elif len(info) == 19:
                panel = Quad(v0=np.array([float(info[1]), float(info[2]), float(info[3])]),
                             v1=np.array([float(info[4]), float(info[5]), float(info[6])]),
                             v2=np.array([float(info[7]), float(info[8]), float(info[9])]),
                             v3=np.array([float(info[10]), float(info[11]), float(info[12])]),
                             n=np.array([float(info[13]), float(info[14]), float(info[15])]),
                             v_c=np.array([float(info[16]), float(info[17]), float(info[18])]),
                             A=float(info[19]),
                             d_max=float(info[20]))

            # Check for zero area
            if abs(panel.A)<1e-10:
                raise IOError("Panel {0} in the mesh has zero area.".format(i))

            # Store
            self.panels.append(panel)
            self.cp[i] = panel.v_c
            self.n[i] = panel.n
            self.dA[i] = panel.A

        # Loop through edges
        self.kutta_edges = []
        for i in range(N_edges):

            # Split line
            info = lines[i+self.N].split()

            # Initialize edge
            edge = KuttaEdge(np.array([float(info[1]), float(info[2]), float(info[3])]),
                             np.array([float(info[4]), float(info[5]), float(info[6])]),
                             [int(info[7]), int(info[8])])

            self.kutta_edges.append(edge)


    def _check_mesh(self, **kwargs):
        # Checks the mesh is appropriate and determines where Kutta condition should exist

        # Get Kutta angle
        theta_K = kwargs.get("kutta_angle", None)

        # Look for adjacent panels where the Kutta condition should be applied
        if theta_K is not None:
            theta_K = np.radians(theta_K)

            if self._verbose:
                print()
                prog = OneLineProgress(self.N, msg="Determining locations to apply Kutta condition")

            # Get panel angles
            with np.errstate(invalid='ignore'):
                theta = np.abs(np.arccos(np.einsum('ijk,ijk->ij', self.n[:,np.newaxis], self.n[np.newaxis,:])))

            # Determine which panels have an angle greater than the Kutta angle
            angle_greater = (theta>theta_K).astype(int)
            i_panels = np.argwhere(np.sum(angle_greater, axis=1).flatten()).flatten()

            # Initialize edge storage
            self.kutta_edges = []

            # Loop through possible combinations
            for i in i_panels:

                j_panels = np.argwhere(angle_greater[i]).flatten()
                for j in j_panels:

                    # Don't repeat
                    if j <= i:
                        continue

                    # Get panels
                    panel_i = self.panels[i]
                    panel_j = self.panels[j]
                    
                    # Determine if we're adjacent
                    v0 = None
                    for vi in panel_i.vertices:
                        for vj in panel_j.vertices:

                            # Check distance
                            d = norm(vi-vj)
                            if d > panel_i.d_max+panel_j.d_max:
                                break # There's no way for them to be touching then

                            elif d<1e-8:

                                # Store first
                                if v0 is None:
                                    v0 = vi

                                # Initialize edge object
                                else:
                                    self.kutta_edges.append(KuttaEdge(v0, vi, [i, j]))
                                    break

                if self._verbose:
                    prog.display()


            self.N_edges = len(self.kutta_edges)
            if self._verbose: print("   {0} Kutta edges detected.".format(self.N_edges), flush=True)

        else:
            self.N_edges = 0


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
            for i, panel in enumerate(self.panels):
                n_vert = panel.vertices.shape[1]
                ind = [x%n_vert for x in range(n_vert+1)]
                ax.plot(panel.vertices[ind,0], panel.vertices[ind,1], panel.vertices[ind,2], 'k-', label='Panel' if i==0 else '')
        
        # Plot centroids
        if kwargs.get("centroids", True):
            for i, panel in enumerate(self.panels):
                ax.plot(panel.v_c[0], panel.v_c[1], panel.v_c[2], 'r.', label='Centroid' if i==0 else '')

        # Plot Kutta edges
        if kwargs.get("kutta_edges", True) and hasattr(self, "kutta_edges"):
            for i, edge in enumerate(self.kutta_edges):
                ax.plot(edge.vertices[:,0], edge.vertices[:,1], edge.vertices[:,2], 'b', label='Kutta Edge' if i==0 else '')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        lims= ax.get_ylim()
        ax.set_xlim3d(lims[0], lims[1])
        ax.set_zlim3d(lims[0], lims[1])
        plt.legend()
        plt.show()


    def export(self, filename):
        """Exports the mesh with all relevant parameters to make reloading faster.

        Parameters
        ----------
        filename : str
            Name of the file to write the mesh to.
        """

        # Open file
        with open(filename, 'w') as export_handle:

            # Write panels
            for panel in self.panels:
                print(str(panel), file=export_handle)

            # Write Kutta edges
            for edge in self.kutta_edges:
                print(str(edge), file=export_handle)
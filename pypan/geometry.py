"""Defines classes for handling basic geometric elements."""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

from pypan.pp_math import vec_norm, norm, vec_inner, inner, vec_cross, cross


class Panel:
    """A base class defining a panel for potential flow simulation."""

    def __init__(self, **kwargs):

        # Get normal vector
        self.n = kwargs.get("n")
        if self.n is None:
            self._calc_normal()
        else:
            # Check normalization
            if norm(self.n) != 0.0:
                self.n = self.n/norm(self.n)

        # Calculate panel plane projection matrix
        self.P_surf = np.identity(3)-np.matmul(self.n, self.n)

        # Determine area
        self.A = kwargs.get("A", None)
        if self.A is None:
            self._calc_area()

        # Determine centroid
        self.v_c = kwargs.get("v_c", None)
        if self.v_c is None:
            self._calc_centroid()

        # Initialize some storage
        self.touching_panels = [] # Panels which share at least one vertex with this panel
        self.abutting_panels = [] # Panels which share two vertices with this panel
        self.touching_panels_not_across_kutta_edge = [] # Panels which share at least one vertex with this panel where those two vertices do not define a Kutta edge
        self.abutting_panels_not_across_kutta_edge = [] # Panels which share two vertices with this panel where those two vertices do not define a Kutta edge
        self.second_abutting_panels_not_across_kutta_edge = [] # Panels which share two vertices with this panel or its abutting panels where those two vertices do not define a Kutta edge
        self.gradient_panels = self.second_abutting_panels_not_across_kutta_edge


    def _calc_normal(self):
        # Calculates the panel unit normal vector
        # Assumes the panel is planar
        d1 = self.vertices[1]-self.vertices[0]
        d2 = self.vertices[2]-self.vertices[1]
        N = cross(d1, d2)
        self.n = N/norm(N)


    def _calc_centroid(self):
        # Calculates the centroid of the panel
        self.v_c = 1.0/self.N*np.sum(self.vertices, axis=0).flatten()


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
        r = points[np.newaxis,:,:]-self.vertices[:,np.newaxis,:]
        r_mag = vec_norm(r)

        # Calculate influence
        v = np.zeros_like(points)
        for i in range(self.N):
            v += vec_cross(r[i-1], r[i])*((r_mag[i-1]+r_mag[i])/(r_mag[i-1]*r_mag[i]*(r_mag[i-1]*r_mag[i]+vec_inner(r[i-1], r[i]))))[:,np.newaxis]

        return 0.25/np.pi*v


    @abstractmethod
    def _calc_area(self):
        pass


    @abstractmethod
    def mirror(self):
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
        self.midpoints = 0.5*(self.vertices+np.roll(self.vertices, 1, axis=0))

        self.N = 4

        super().__init__(**kwargs)

        # Set up local coordinate transformation
        self.A_t = np.zeros((3,3))
        self.A_t[0] = self.midpoints[1]-self.midpoints[0]
        self.A_t[0] /= norm(self.A_t[0])
        self.A_t[1] = cross(self.n, self.A_t[0])
        self.A_t[2] = self.n


    def _calc_area(self):
        # Calculates the panel area from the two constituent triangles
        self.A = 0.5*norm(cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]))
        self.A += 0.5*norm(cross(self.vertices[2]-self.vertices[0], self.vertices[3]-self.vertices[0]))


    def _calc_normal(self):
        # Calculates the normal based off of the edge midpoints
        self.n = cross(self.midpoints[1]-self.midpoints[0], self.midpoints[2]-self.midpoints[1])
        self.n /= norm(self.n)


class Tri(Panel):
    """A triangular panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")

        self.N = 3

        super().__init__(**kwargs)

        # Set up local coordinate transformation
        self.A_t = np.zeros((3,3))
        self.A_t[0] = self.vertices[1]-self.vertices[0]
        self.A_t[0] /= norm(self.A_t[0])
        self.A_t[1] = cross(self.n, self.A_t[0])
        self.A_t[2] = self.n


    def _calc_area(self):
        # Calculates the panel area
        self.A = 0.5*norm(cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]))


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
                     self.A)
        return s


class KuttaEdge:
    """A class for defining an edge segment at which the Kutta condition is applied. These are used
    for the horseshoe vortex type wake.

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


    def get_vortex_influence(self, points, u_inf):
        """Determines the velocity vector induced by this edge at arbitrary
        points, assuming a horseshoe vortex is shed from this edge.

        Parameters
        ----------
        points : ndarray
            An array of points where the first index is the point index and 
            the second index is the coordinate.

        u_inf : ndarray
            Freestream direction vectors. Same shape as points.

        Returns
        -------
        ndarray
            The velocity vector induced at each point.
        """

        # Determine displacement vectors
        r0 = points-self.vertices[0,:]
        r1 = points-self.vertices[1,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)[:,np.newaxis]
        r1_mag = vec_norm(r1)[:,np.newaxis]

        # Calculate influence of bound segment
        v_01 = ((r0_mag+r1_mag)*vec_cross(r0, r1))/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1)[:,np.newaxis]))

        # Calculate influence of trailing segments
        v_0_inf = vec_cross(u_inf, r0)/(r0_mag*(r0_mag-vec_inner(u_inf, r0)[:,np.newaxis]))
        v_1_inf = vec_cross(u_inf, r1)/(r1_mag*(r1_mag-vec_inner(u_inf, r1)[:,np.newaxis]))

        return 0.25/np.pi*(-v_0_inf+v_01+v_1_inf)